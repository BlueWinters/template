
import logging
import os, yaml
from PyQt5 import QtCore, QtWidgets, QtGui, Qt



class ViewerFormMain(object):
    def __init__(self, form:QtWidgets.QWidget):
        self.form = form
        self.setupUi(form)
        self.setupMsg()
        self.setupFcn()
        self.setupVars()
        self.setupMisc()

    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1561, 913)
        self.graphicsViewInput = QtWidgets.QGraphicsView(Form)
        self.graphicsViewInput.setGeometry(QtCore.QRect(20, 20, 751, 831))
        self.graphicsViewInput.setObjectName("graphicsViewInput")
        self.graphicsViewOutput = QtWidgets.QGraphicsView(Form)
        self.graphicsViewOutput.setGeometry(QtCore.QRect(790, 20, 751, 831))
        self.graphicsViewOutput.setObjectName("graphicsViewOutput")
        self.toolButtonOpen = QtWidgets.QToolButton(Form)
        self.toolButtonOpen.setEnabled(False)
        self.toolButtonOpen.setGeometry(QtCore.QRect(130, 870, 91, 31))
        self.toolButtonOpen.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)
        self.toolButtonOpen.setObjectName("toolButtonOpen")
        self.pushButtonOption = QtWidgets.QPushButton(Form)
        self.pushButtonOption.setEnabled(False)
        self.pushButtonOption.setGeometry(QtCore.QRect(460, 870, 93, 31))
        self.pushButtonOption.setObjectName("pushButtonOption")
        self.pushButtonClear = QtWidgets.QPushButton(Form)
        self.pushButtonClear.setEnabled(False)
        self.pushButtonClear.setGeometry(QtCore.QRect(350, 870, 93, 31))
        self.pushButtonClear.setObjectName("pushButtonClear")
        self.horizontalSliderStrokeSize = QtWidgets.QSlider(Form)
        self.horizontalSliderStrokeSize.setEnabled(False)
        self.horizontalSliderStrokeSize.setGeometry(QtCore.QRect(570, 876, 201, 20))
        self.horizontalSliderStrokeSize.setMinimum(1)
        self.horizontalSliderStrokeSize.setMaximum(20)
        self.horizontalSliderStrokeSize.setProperty("value", 20)
        self.horizontalSliderStrokeSize.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSliderStrokeSize.setObjectName("horizontalSliderStrokeSize")
        self.pushButtonInit = QtWidgets.QPushButton(Form)
        self.pushButtonInit.setEnabled(True)
        self.pushButtonInit.setGeometry(QtCore.QRect(20, 870, 93, 31))
        self.pushButtonInit.setObjectName("pushButtonInit")
        self.toolButtonRun = QtWidgets.QToolButton(Form)
        self.toolButtonRun.setEnabled(False)
        self.toolButtonRun.setGeometry(QtCore.QRect(240, 870, 91, 31))
        self.toolButtonRun.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)
        self.toolButtonRun.setObjectName("toolButtonRun")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "matte"))
        self.toolButtonOpen.setText(_translate("Form", "open"))
        self.pushButtonOption.setText(_translate("Form", "option"))
        self.pushButtonClear.setText(_translate("Form", "clear"))
        self.pushButtonInit.setText(_translate("Form", "init"))
        self.toolButtonRun.setText(_translate("Form", "all"))

    def setupMsg(self):
        self.pushButtonInit.clicked.connect(self.pushButtonInitClicked)
        self.toolButtonOpen.clicked.connect(self.toolButtonOpenClicked)
        self.toolButtonRun.clicked.connect(self.toolButtonRunClicked)
        self.pushButtonOption.clicked.connect(self.pushButtonOptionClicked)
        self.pushButtonClear.clicked.connect(self.pushButtonClearClicked)
        self.graphicsViewInput.contextMenuEvent = self.graphicsViewInputContextMenuEvent

    def setupFcn(self):
        from tools.gui.fcn.fcn_viewer import FcnViewDisplay, FcnViewerSyn
        self.fcnViewDisIn = FcnViewDisplay(self.graphicsViewInput)
        self.fcnViewDisOt = FcnViewDisplay(self.graphicsViewOutput)
        self.fcnViewerSyn = FcnViewerSyn(view_list=[self.graphicsViewInput, self.graphicsViewOutput])
        from tools.gui.fcn.fcn_button import FcnButtonOpen
        self.fcnButtonOpen = FcnButtonOpen(self.form, self.toolButtonOpen)

    def setupVars(self):
        from project.libcore import LibCore
        self.core = LibCore()
        self.data = self.loadConfig('tools/gui/config.yaml')

    def setupMisc(self):
        menu_open = QtWidgets.QMenu()
        for name in self.core.object: menu_open.addAction(
            name, self.toolButtonRunClickedSubmenu(name))
        menu_open.addAction('all', self.toolButtonRunClickedSubmenu('all'))
        self.toolButtonRun.setMenu(menu_open)

    """
    """
    def pushButtonInitClicked(self):
        self.core.initialize()
        self.toolButtonOpen.setEnabled(True)
        self.pushButtonOption.setEnabled(True)
        self.printMessage('normal', 'init finish', False)

    def toolButtonOpenClicked(self):
        try:
            self.createSession(self.fcnButtonOpen.clickButton())
        except NotImplementedError:
            self.printMessage('warning', 'select local or link', False)

    def toolButtonRunClicked(self):
        name = self.toolButtonRun.text()
        if self.graphicsViewInput.scene() is not None:
            from .misc.trimap import TrimapGenerator
            tri = self.fcnViewDisIn.getCanvasResults()['stroke'][0]
            tri = TrimapGenerator.format(tri)
            if name == 'all':
                from .misc.container import ContainerImage
                mat = self.core.pipeline(self.data['image'].bgr, tri)
                method = '/'.join(list(mat.keys()))
                vis = ContainerImage.concatenate([mat[m] for m in mat])
                self.printMessage('normal', 'estimate alpha from {}'.format(method), False)
                self.fcnViewDisOt.showSubWindow('all', vis)
            else:
                mat = self.core.pipeline(self.data['image'].bgr, tri, model=name)
                self.printMessage('normal', 'estimate alpha from {}'.format(name), False)
                self.fcnViewDisOt.displayViewInput(mat[name])

    def toolButtonRunClickedSubmenu(self, name):
        def setText():
            self.toolButtonRun.setText(name)
        return setText

    def pushButtonOptionClicked(self):
        pass

    def pushButtonClearClicked(self):
        self.fcnViewDisIn.clearMarks()

    def graphicsViewInputContextMenuEvent(self, event:QtGui.QContextMenuEvent):
        def getQAction(title, function, checked):
            action = QtWidgets.QAction(title, menu)
            action.triggered.connect(function)
            action.setCheckable(True), action.setChecked(checked)
            return action

        menu = QtWidgets.QMenu()
        if self.graphicsViewInput.scene() is not None:
            flag, x, y = self.fcnViewDisIn.mapToSceneCoordinate(event.x(), event.y())
            if flag is True:
                # complex menu
                sub_menu_trimap = menu.addMenu('triple-map')
                sub_menu_trimap.addAction(getQAction('show', self.ctxMenuActionTriMapSubMenuShow, False))
                sub_menu_trimap.addSection('from')
                sub_menu_trimap.addAction(getQAction('from-file', self.ctxMenuActionTriMapSubMenuFromFile, False))
                sub_menu_trimap.addSection('mark')
                is_foreground = bool(self.data['mark']['current'] == 'foreground')
                sub_menu_trimap.addAction(getQAction('foreground', lambda: self.setMarkType('foreground'), is_foreground))
                sub_menu_trimap.addAction(getQAction('transition', lambda: self.setMarkType('transition'), not is_foreground))
                sub_menu_select = menu.addMenu('select-as')
                sub_menu_select.addAction(getQAction('show', self.ctxMenuActionSelectAsSubMenuShow(x, y), False))
                sub_menu_select.addAction(getQAction('foreground', self.ctxMenuActionSelectAsSubMenuForeground(x, y), False))
                sub_menu_select.addAction(getQAction('transition', self.ctxMenuActionSelectAsSubMenuTransition(x, y), False))
                # single menu
                menu.addAction('super-pixel', self.ctxMenuActionSuperpixel)
        else:
            menu.addAction('auto', self.ctxMenuActionAuto)
        menu.exec_(Qt.QCursor.pos())

    """
    """
    def loadConfig(self, yaml_path:str):
        if os.path.exists(yaml_path) is False:
            print('load config fail: {}'.format(yaml_path))
            raise FileNotFoundError
        config = yaml.load(open(yaml_path, 'r'), Loader=yaml.Loader)
        return config

    def printMessage(self, header, message:str, msg_box:bool=False):
        line = '{}: {}'.format(header, message)
        print(line)
        # self.plainTextEditMsg.insertPlainText(line+'\n')
        # self.plainTextEditMsg.moveCursor(QtGui.QTextCursor.End)
        if msg_box is True:
            QtWidgets.QMessageBox.warning(
                self.form, header, message, QtWidgets.QMessageBox.Yes)

    def createSession(self, path):
        try:
            from .misc.container import ContainerImage
            self.data['image'] = ContainerImage(path=path)
            self.printMessage('normal', 'load image success', False)
            self.fcnViewDisIn.displayViewInput(self.data['image'].rgb)
            self.fcnViewDisOt.invalidImage()
            self.setMarkType()
            self.enableComponent(True)
        except AssertionError:
            self.printMessage('warning', 'load image fail', False)
            self.enableComponent(False)

    def enableComponent(self, flag:bool):
        self.toolButtonRun.setEnabled(flag)
        self.pushButtonClear.setEnabled(flag)
        self.horizontalSliderStrokeSize.setEnabled(flag)

    """
    """
    def ctxMenuActionAuto(self):
        self.createSession('data/matte/image/net.png')
        self.inputTriMap('data/matte/trimap/net.png')

    def ctxMenuActionTriMapSubMenuShow(self):
        scene = self.graphicsViewInput.scene()
        if isinstance(scene, QtWidgets.QGraphicsScene):
            from .misc.trimap import TrimapGenerator
            self.printMessage('normal', 'show triple-map', False)
            mark = self.data['mark']
            tri = self.fcnViewDisIn.getCanvasResults()['stroke'][0]
            rgb = TrimapGenerator.visual(tri, mark['index'], mark['color'])
            self.fcnViewDisIn.showSubWindow('trimap', image=rgb)

    def ctxMenuActionTriMapSubMenuFromFile(self):
        scene = self.graphicsViewInput.scene()
        if isinstance(scene, QtWidgets.QGraphicsScene):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.form, 'open a file', '.', filter='*.png;*.bmp')
            self.inputTriMap(path)

    def ctxMenuActionSelectAsSubMenuShow(self, x, y):
        def selectRegion():
            region = self.getSelectRegion(x, y, sign=(127, 0))
            self.fcnViewDisIn.showSubWindow('region', image=region)
        return selectRegion

    def ctxMenuActionSelectAsSubMenuForeground(self, x, y):
        def markSelectRegionAs():
            index = self.data['mark']['index']['foreground']
            region = self.getSelectRegion(x, y, sign=[index, 0])
            self.fcnViewDisIn.inputMark(mask=region, mode='select')
        return markSelectRegionAs

    def ctxMenuActionSelectAsSubMenuTransition(self, x, y):
        def markSelectRegionAs():
            index = self.data['mark']['index']['transition']
            region = self.getSelectRegion(x, y, sign=[index, 0])
            self.fcnViewDisIn.inputMark(mask=region, mode='select')
        return markSelectRegionAs

    def ctxMenuActionSuperpixel(self):
        scene = self.graphicsViewInput.scene()
        if isinstance(scene, QtWidgets.QGraphicsScene):
            from project.segment.libsuperpixel import LibSuperPixel
            self.printMessage('normal', 'show super-pixel', False)
            bgr = LibSuperPixel.split(bgr=self.data['image'].bgr)
            self.fcnViewDisIn.showSubWindow('superpixel', image=bgr[:, :, ::-1])

    """
    """
    def getSelectRegion(self, x, y, sign=(1, 0)):
        from .misc.trimap import TrimapGenerator
        self.printMessage('normal', 'select region', False)
        tri = self.fcnViewDisIn.getCanvasResults()['stroke'][0]
        xx, yy = self.fcnViewDisIn.mapToImageCoordinate(x, y)
        region = TrimapGenerator.select(tri, xx, yy, sign=sign)
        return region

    def setMarkType(self, name=None):
        mark = self.data['mark']
        assert name in mark['color'] or name is None
        current = mark['current'] = name if name is not None else mark['current']
        self.fcnViewDisIn.setCanvasColor(color=mark['color'][current])

    def inputTriMap(self, path):
        if os.path.exists(path):
            from .misc.trimap import TrimapGenerator
            self.printMessage('normal', 'input triple-map file', False)
            self.pushButtonClearClicked()
            tri = TrimapGenerator.from_file(path)
            self.fcnViewDisIn.inputMark(mask=tri, mode='overwrite')