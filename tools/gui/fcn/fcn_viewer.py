
import numpy as np
from typing import List, Tuple
from PyQt5 import QtCore, QtWidgets, Qt
from ..misc.scene import QGraphicsSceneCanvas


class FcnViewDisplay(object):
    def __init__(self, view:QtWidgets.QGraphicsView):
        self.view = view
        self.sub = dict()

    def displayViewInput(self, image:np.ndarray):
        h, w = self.view.height(), self.view.width()
        scene = QGraphicsSceneCanvas(parent=self, size=(h, w), image=image)
        self.view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.view.setScene(scene)
        self.view.show()

    def getCanvasScene(self) -> QGraphicsSceneCanvas:
        scene = self.view.scene()
        assert isinstance(scene, QGraphicsSceneCanvas)
        return scene

    def setCanvasColor(self, color:Tuple[int, int, int]):
        self.getCanvasScene().setColor(color)

    def getCanvasResults(self, **kwargs):
        return self.getCanvasScene().getResults(**kwargs)

    def mapToSceneCoordinate(self, vx:int, vy:int):
        scene = self.getCanvasScene()
        sw, sh = int(scene.width()), int(scene.height())
        point = self.view.mapToScene(vx, vy)
        sx, sy = int(point.x()), int(point.y())
        fw = bool(0 <= sx < sw)
        fh = bool(0 <= sy < sh)
        return bool(fw & fh), sx, sy

    def mapToImageCoordinate(self, x:int, y:int):
        return self.getCanvasScene().mapToImageCoordinate(x, y)

    def inputMark(self, *args, **kwargs):
        self.getCanvasScene().inputMark(args, **kwargs)

    def clearMarks(self):
        self.getCanvasScene().clearMark()

    def invalidImage(self):
        self.view.invalidateScene()

    def showSubWindow(self, name:str, image:np.ndarray):
        if name not in self.sub:
            self.sub[name] = QtWidgets.QGraphicsView()
            ww, hh = self.view.width(), self.view.height()
            self.sub[name].setGeometry(QtCore.QRect(10, 10, ww, hh))
            self.sub[name].setWindowTitle(name)
        # display position
        view = self.sub[name]
        pos = self.view.pos()
        pos = self.view.parent().mapToGlobal(pos)
        view.move(pos)
        view.setFocusPolicy(QtCore.Qt.ClickFocus)
        # update scene
        h, w = view.height(), view.width()
        scene = QGraphicsSceneCanvas(parent=self, size=(h, w), image=image)
        view.setScene(scene)
        view.setWindowModality(QtCore.Qt.ApplicationModal)
        view.show()



class FcnViewerSyn(object):
    def __init__(self, view_list:List[QtWidgets.QGraphicsView]):
        self.view_list = view_list
        self.setupMsg()

    def setupMsg(self):
        for view in self.view_list:
            assert isinstance(view, QtWidgets.QGraphicsView)
            view.wheelEvent = self.graphicsViewAllWheelEvent
            view.scrollContentsBy = self.graphicsViewAllScrollContentsBy(view)

    def graphicsViewAllWheelEvent(self, event:Qt.QWheelEvent):
        for view in self.view_list:
            assert isinstance(view, QtWidgets.QGraphicsView)
            QtWidgets.QGraphicsView.wheelEvent(view, event)

    def graphicsViewAllScrollContentsBy(self, view:QtWidgets.QGraphicsView):
        def graphicsViewScrollContentsBy(dx, dy):
            QtWidgets.QGraphicsView.scrollContentsBy(view, dx, dy)
            value_hor = view.horizontalScrollBar().value()
            value_ver = view.verticalScrollBar().value()
            for v in self.view_list:
                v.horizontalScrollBar().setValue(value_hor)
                v.verticalScrollBar().setValue(value_ver)
                v.viewport().update()
        return graphicsViewScrollContentsBy
