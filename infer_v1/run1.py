from ourUI import Ui_MainWindow
from PyQt5 import QtWidgets
import sys

app = QtWidgets.QApplication(sys.argv)
ui = Ui_MainWindow()
ui.show()
sys.exit(app.exec_())

