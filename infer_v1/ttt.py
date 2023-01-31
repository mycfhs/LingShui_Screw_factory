# import numpy as np
# from scipy.optimize import minimize
# import matplotlib.pyplot as plt
# npoints = 75
# theta = np.random.uniform(0, 2*np.pi, size=npoints)
# a, b = (3, 5)
# initial_parameter_guess = (2.5, 6)
# xnoise = np.random.uniform(-1, 1, size=theta.size)
# ynoise = np.random.uniform(-1, 1, size=theta.size)
# x = a * np.cos(theta)
# xn = x + xnoise
# y = b * np.sin(theta)
# yn = y + ynoise
#
# def get_residuals(prms, x, y):
#     """ """
#     return 1 - ((x/prms[0])**2 + (y/prms[1])**2)
#
# def f_error(prms, x, y):
#     """ """
#     resid = get_residuals(prms, x, y)
#     return np.sum(np.square(resid))
# result = minimize(f_error, x0=initial_parameter_guess,args=(xn, yn))
#
# yf = result.x[0] * x + result.x[1]
#
# fig, ax = plt.subplots()
# ax.scatter(x, yn, color='b', marker='.')
# ax.plot(x, yf, color='r', alpha=0.5)
# ax.grid(color='k', alpha=0.3, linestyle=':')
# plt.show()
# plt.close(fig)
# print(result)
