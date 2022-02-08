
import sys
from PyQt5 import QtCore, QtWidgets



if __name__ == '__main__':
    from tools.gui.form import ViewerFormMain
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = ViewerFormMain(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
