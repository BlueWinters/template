
import os
import requests
from typing import Callable
from PyQt5 import QtCore, QtGui, QtWidgets, Qt


class ContainerQtButtonOpen(object):
    def __init__(self, form:QtWidgets.QWidget, button:QtWidgets.QToolButton):
        self.form = form
        self.button = button
        self.setupMsg()
        self.setupVar()

    def setupMsg(self):
        # sub menu
        menu_open = QtWidgets.QMenu()
        menu_open.addAction('local', self.toolButtonOpenClickedSubmenuLocalFcn)
        menu_open.addAction('link', self.toolButtonOpenClickedSubmenuLinkFcn)
        self.button.setMenu(menu_open)

    def setupVar(self):
        self.openStr = self.button.text()  # 'open'
        self.openFcn = None
        self.openLocalStr = 'local'
        self.openLinkStr = 'link'
        self.openFilterStr = '*.png;*.jpg;*.bmp'

    def toolButtonOpenClickedSubmenuLocalFcn(self):
        self.button.setText(self.openLocalStr)
        self.openFcn = self.fcnOpenClickedLocal

    def toolButtonOpenClickedSubmenuLinkFcn(self):
        self.button.setText(self.openLinkStr)
        self.openFcn = self.fcnOpenClickedLink

    def fcnOpenClickedLocal(self):
        default_path = 'data/matte' #self.default_config['open_local_path']
        default_path = default_path if os.path.exists(default_path) else ''
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.form, 'open a file', default_path, filter=self.openFilterStr)
        return path
        # self.default_config['open_local_path'] = os.path.split(path)[0] if os.path.exists(path) else default_path

    def fcnOpenClickedLink(self):
        return None
        # url, ok = QtWidgets.QInputDialog.getText(self.form, 'URL', 'input image url:')
        # if not (len(url) > 0 and ok is True):
        #     return
        # try:
        #     receive = requests.get(url, timeout=10)
        #     if receive.status_code == 200:
        #         bgr = cv2.imdecode(np.frombuffer(receive.content, dtype=np.uint8), cv2.IMREAD_COLOR)
        #         self.createSession(data=ContainerImage(bgr=bgr, name=url))
        #     else:
        #         self.printMessage('error', 'http requests code({})'.format(receive.status_code))
        # except requests.exceptions.Timeout as e:
        #     self.printMessage('error', 'http requests timeout, code({})'.format(e.response))

    """
    """
    def enableButton(self, flag:bool):
        self.button.setEnabled(flag)

    def clickButton(self):
        text = self.button.text()
        if text == self.openStr:
            raise NotImplementedError
        return self.openFcn()

    def setFilterString(self, filter:str):
        self.openFilterStr = filter
