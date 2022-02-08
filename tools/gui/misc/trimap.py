
import os
import cv2
import numpy as np
import skimage.measure as measure
from scipy import signal



class TrimapGenerator:
    """
    """
    TrimapLabel = [0, 1, 2]
    TriMapLabelDict = {
        'foreground'
    }

    """
    """
    @staticmethod
    def visual(tri:np.ndarray, index, color):
        # r for transition, b for foreground
        assert len(tri.shape) == 2
        assert len(color) == len(index)
        mask = np.zeros(shape=(*tri.shape, 3), dtype=np.uint8)
        mask[tri == index['foreground'], ...] = color['foreground']
        mask[tri == index['transition'], ...] = color['transition']
        return mask

    @staticmethod
    def format(image:np.ndarray):
        if len(image.shape) == 3:
            mask = np.zeros(shape=image.shape[:2], dtype=np.uint8)
            mask[image[:, :, 0] > 0] = 2
            mask[image[:, :, 2] > 0] = 1
            return mask
        if len(image.shape) == 2:
            tri = np.zeros(shape=image.shape, dtype=np.uint8)
            label = sorted(list(np.unique(image.astype(np.uint8))))
            assert 1 < len(label) <= 3
            if TrimapGenerator.TrimapLabel[0] not in label:
                label.insert(0, 0)
            for n, value in enumerate(label):
                tri[image == value] = TrimapGenerator.TrimapLabel[n]
            return tri
        raise NotImplementedError

    @staticmethod
    def select(tri:np.ndarray, x:int, y:int, sign:(int,int)):
        h, w = tri.shape
        assert 0 <= x < w and 0 <= y < h
        region, num = measure.label((tri+1).astype(np.uint8), connectivity=2, return_num=True)
        return np.where(region == int(region[y,x]), sign[0], sign[1]).astype(np.uint8)

    @staticmethod
    def from_file(path:str):
        assert os.path.exists(path)
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        return TrimapGenerator.format(image)

    """
    """
    def __init__(self):
        self.filter_tri, self.flag_tri = self.calculate_filter(k=11)
        self.filter_pre, self.flag_pre = self.calculate_filter(k=3)

    def calculate_filter(self, k: int = 11):
        cx, cy = k // 2, k // 2
        center = np.reshape(np.array([cx, cy], dtype=np.int32), (1, 1, 2))
        xx = np.arange(0, k, 1)
        yy = np.arange(0, k, 1)
        x, y = np.meshgrid(xx, yy)
        mesh = np.stack([x, y], axis=2).astype(np.int32)
        ds = np.square(mesh - center)
        distance = np.sum(ds, axis=2)
        filter = np.where(distance < 150, 1, 0).astype(np.int32)
        return filter, np.sum(filter)

    def calculate_tri(self, mask, filter, flag):
        bdr = signal.convolve2d(mask, filter, mode='same', boundary='symm')
        bdr = np.where((bdr > 0) & (bdr < flag), 1, 0)
        output = np.zeros_like(mask)  # background is 0
        output[mask == 1] = 2
        output[bdr == 1] = 1
        return output

    def connectivity_process(self, tri):
        assert len(np.unique(tri) == 3)
        mask = np.where(tri > 0, np.ones_like(tri), np.zeros_like(tri))
        region, n = measure.label(mask, connectivity=2, return_num=True)
        props = measure.regionprops(region)
        num_pix = []
        for ia in range(len(props)):
            num_pix += [props[ia].area]
        max_num = max(num_pix)
        index = num_pix.index(max_num)
        label = props[index].label
        return np.where(region == label, np.ones_like(region), np.zeros_like(region)).astype(np.uint8)

    def calculate_tri_from_seg(self, seg):
        bdr_pre = self.calculate_tri(seg, self.filter_pre, self.flag_pre)
        new_seg = self.connectivity_process(bdr_pre) * seg
        bdr_out = self.calculate_tri(new_seg, self.filter_tri, self.flag_tri)
        return bdr_out

    def __call__(self, matting):
        segment = np.where(matting > 0, np.ones_like(matting), np.zeros_like(matting))
        return self.calculate_tri_from_seg(segment)