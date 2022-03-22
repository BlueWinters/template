
import sys
from PyQt5 import QtCore, QtWidgets


def execute_gui():
    from utils.gui.form import ViewerFormMain
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = ViewerFormMain(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())




if __name__ == '__main__':
    execute_gui()

