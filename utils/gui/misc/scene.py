
import os
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from .canvas import Canvas



class QGraphicsSceneCanvas(QtWidgets.QGraphicsScene):
    def __init__(self, *args, **kwargs):
        super(QGraphicsSceneCanvas, self).__init__(*args)
        # data structure
        self.size = kwargs['size']
        self.rgb = self._formatImage(kwargs['image'])
        # canvas
        self.canvas = Canvas(size=self.size, rgb=self.rgb)
        # the first handle
        self.setMark(kwargs['mark'])
        # first display
        self._displayImage(self.canvas.update())

    def _formatImage(self, image:np.ndarray):
        rgb = image if len(image.shape) == 3 else \
            cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return rgb.copy()

    def _displayImage(self, image:np.ndarray):
        from PyQt5.QtWidgets import QGraphicsPixmapItem
        from PyQt5.QtGui import QPixmap, QImage
        rgb = image if len(image.shape) == 3 else \
            cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        h, w, c = rgb.shape
        img = QImage(bytes(rgb.data), w, h, w * c, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        item = QGraphicsPixmapItem(pix)
        self.clear()
        self.setSceneRect(0, 0, w, h)
        self.addItem(item)

    def wheelEvent(self, event:QtWidgets.QGraphicsSceneWheelEvent):
        if event.modifiers() == QtCore.Qt.ControlModifier:
            sign = 1 if event.delta() > 0 else -1
            image_vis = self.canvas.wheel_event(sign)
            self._displayImage(image_vis)

    def mousePressEvent(self, event):
        event_list = list()
        if event.buttons() == QtCore.Qt.LeftButton:
            event_list.append('left-button')
        if event.buttons() == QtCore.Qt.RightButton:
            event_list.append('right-button')
        x, y = event.scenePos().x(), event.scenePos().y()
        image_vis = self.canvas.mouse_press_event(x, y, event_list)
        self._displayImage(image_vis)

    def mouseMoveEvent(self, event):
        event_list = list()
        if event.buttons() == QtCore.Qt.LeftButton:
            event_list.append('left-button')
        if event.buttons() == QtCore.Qt.RightButton:
            event_list.append('right-button')
        x, y = event.scenePos().x(), event.scenePos().y()
        image_vis = self.canvas.mouse_move_event(x, y, event_list)
        self._displayImage(image_vis)

    def mouseReleaseEvent(self, event):
        event_list = list()
        if event.button() == QtCore.Qt.LeftButton:
            event_list.append('left-button')
        x, y = event.scenePos().x(), event.scenePos().y()
        image_vis = self.canvas.mouse_release_event(x, y, event_list)
        self._displayImage(image_vis)

    # def mouseDoubleClickEvent(self, event):
    #     event_list = list()
    #     if event.button() == QtCore.Qt.LeftButton:
    #         event_list.append('left-button')
    #     x, y = event.scenePos().x(), event.scenePos().y()
    #     image_vis = self.canvas.mouse_double_click_event(x, y, event_list)
    #     self._displayImage(image_vis)

    def keyPressEvent(self, event:QtGui.QKeyEvent):
        event_list = list()
        if event.key() == QtCore.Qt.Key_Escape:
            event_list.append('key-escape')
        image_vis = self.canvas.mouse_key_press_event(event_list)
        self._displayImage(image_vis)

    """
    """
    def setScale(self, scale:float, update:bool=True):
        image_vis = self.canvas.set_scale(scale, update)
        if update is True:
            self._displayImage(image_vis)

    def getScale(self):
        return self.canvas.get_scale()

    def setMark(self, handle):
        self.canvas.change_handle(handle)

    def getMark(self) -> str:
        return self.canvas.get_mark()

    def setColor(self, color):
        self.canvas.set_color(color)

    def getResults(self, *args, **kwargs):
        return self.canvas.get_results(**kwargs)

    def setRadiusSize(self, radius):
        self.canvas.set_parameters(radius=radius)

    def resetRGBImage(self, rgb:np.ndarray):
        self._displayImage(self.canvas.reset_image(rgb=rgb))

    def inputMark(self, *args, **kwargs):
        image_vis = self.canvas.input_once_event(*args, **kwargs)
        self._displayImage(image_vis)

    def clearMark(self):
        type = self.canvas.clear_marks()
        self.canvas.change_handle(type)
        self._displayImage(self.canvas.update())

    def mapToImageCoordinate(self, x:int, y:int):
        sh, sw = self.height(), self.width()
        ih, iw = self.rgb.shape[:2]
        xx = int(x / sw * iw + 0.5)
        yy = int(y / sh * ih + 0.5)
        return xx, yy
