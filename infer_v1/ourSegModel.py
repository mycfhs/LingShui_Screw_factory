from yolact_edge.data import COLORS, cfg, set_cfg
from yolact_edge.yolact import Yolact
from yolact_edge.utils.augmentations import BaseTransform, FastBaseTransform
from yolact_edge.utils import timer
from yolact_edge.utils.functions import SavePath
from yolact_edge.layers.output_utils import postprocess, undo_image_transformation
from yolact_edge.utils.tensorrt import convert_to_tensorrt

import torch
import torch.backends.cudnn as cudnn
import argparse
import time
import random

from collections import defaultdict

from ourMeasureFun import *


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')
    parser.add_argument('--trained_model',
                        default='models/yolact_edge_mobilenetv2_64_2125.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--top_k', default=10, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')

    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')
    parser.add_argument('--eval_stride', default=5, type=int,
                        help='The default frame eval stride.')
    parser.add_argument('--config', default='yolact_edge_mobilenetv2_config',
                        help='The config object to use.')
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--fast_eval', default=False, dest='fast_eval', action='store_true',
                        help='Skip those warping frames when there is no GT annotations.')
    parser.add_argument('--deterministic', default=False, dest='deterministic', action='store_true',
                        help='Whether to enable deterministic flags of PyTorch for deterministic results.')
    parser.add_argument('--seed', default=None, type=int,
                        help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--score_threshold', default=0.6, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
    parser.add_argument('--yolact_transfer', dest='yolact_transfer', action='store_true',
                        help='Split pretrained FPN weights to two phase FPN (for models trained by YOLACT).')
    parser.add_argument('--coco_transfer', dest='coco_transfer', action='store_true',
                        help='[Deprecated] Split pretrained FPN weights to two phase FPN (for models trained by YOLACT).')
    parser.add_argument('--drop_weights', default=None, type=str,
                        help='Drop specified weights (split by comma) from existing model.')
    parser.add_argument('--calib_images', default=None, type=str,
                        help='Directory of images for TensorRT INT8 calibration, for explanation of this field, please refer to `calib_images` in `data/config.py`.')
    parser.add_argument('--trt_batch_size', default=1, type=int,
                        help='Maximum batch size to use during TRT conversion. This has to be greater than or equal to the batch size the model will take during inferece.')
    parser.add_argument('--disable_tensorrt', default=False, dest='disable_tensorrt', action='store_true',
                        help='Don\'t use TensorRT optimization when specified.')
    parser.add_argument('--use_fp16_tensorrt', default=True, dest='use_fp16_tensorrt', action='store_true',
                        help='This replaces all TensorRT INT8 optimization with FP16 optimization when specified.')
    parser.add_argument('--use_tensorrt_safe_mode', default=True, dest='use_tensorrt_safe_mode', action='store_true',
                        help='This enables the safe mode that is a workaround for various TensorRT engine issues.')

    parser.set_defaults(mask_proto_debug=False, crop=True)

    global args
    args = parser.parse_args(argv)

    if args.seed is not None:
        random.seed(args.seed)


iou_thresholds = [x / 100 for x in range(50, 100, 5)]
color_cache = defaultdict(lambda: {})


def display_and_measure(dets_out, img, time_start, mask_alpha=0.45):

    img_gpu = img / 255.0
    h, w, _ = img.shape

    with timer.env('Postprocess'):
        t = postprocess(dets_out, w, h, visualize_lincomb=args.display_lincomb,
                        crop_masks=args.crop,
                        score_threshold=args.score_threshold)
        torch.cuda.synchronize()

    with timer.env('Copy'):
        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            raw_masks = t[3][:args.top_k]
        classes, scores, boxes = [x[:args.top_k].cpu().numpy() for x in t[:3]]

    num_dets_to_consider = min(args.top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < args.score_threshold:
            num_dets_to_consider = j
            break

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (j * 5) % len(COLORS)

        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]

            color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice

    if cfg.eval_mask_branch:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = raw_masks[:num_dets_to_consider, :, :, None]
        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat(
            [get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1

        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider - 1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()



    for j in reversed(range(num_dets_to_consider)):
        x1, y1, x2, y2 = boxes[j, :]
        color = get_color(j)
        score = scores[j]

        cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

        text_str = '%s: %.2f' % (cfg.dataset.class_names[classes[j]], score)

        text_w, text_h = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)[0]

        cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
        cv2.putText(img_numpy, text_str, (x1, y1 - 3), cv2.FONT_HERSHEY_DUPLEX, 0.6, [255, 255, 255], 1,
                    cv2.LINE_AA)

    fps = time.time() - time_start
    cv2.putText(img_numpy, 'FPS:' + str(1 / fps), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return img_numpy, raw_masks, classes, scores, boxes


def load_model():
    parse_args()
    set_cfg(args.config)

    with torch.no_grad():
        cudnn.benchmark = True
        cudnn.fastest = True
        if args.deterministic:
            cudnn.deterministic = True
            cudnn.benchmark = False
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        net = Yolact(training=False)
        net.load_weights(args.trained_model, args=args)
        net.eval()
        convert_to_tensorrt(net, cfg, args, transform=BaseTransform())

        net.detect.use_fast_nms = args.fast_nms
        cfg.mask_proto_debug = args.mask_proto_debug
        net = net.cuda()
        return net


if __name__ == '__main__':
    from Camara import RealsenseCamera, Camera
    # if True:
    try:
        net = load_model()
        cap = RealsenseCamera(True, 3)
        # cap = Camera(1)
        cap.start()
        while True:
            frame, dist = cap.read()

            frame = torch.from_numpy(frame).cuda().float()

            time_start = time.time()
            preds = net(FastBaseTransform()(frame.unsqueeze(0)),
                        extras={"backbone": "full", "interrupt": False,
                                "keep_statistics": False,
                                "moving_statistics": None})["pred_outs"]
            display_and_measure(preds, frame)
    except:
        print('Occur Error! Please check!')
        pass
    finally:
        cap.stop()
"""
source activate yol
python inf.py
"""