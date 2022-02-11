
import numpy as np
from core.engine import *


class LibFGI:
    def __init__(self, config):
        self.config = config
        self.engine = create_engine(config['engine'])
        self.max_size = 512+256

    def __del__(self):
        logging.warning('delete engine LibFGI')

    def _assert(self, bgr, tri):
        assert len(bgr.shape) == 3
        assert len(tri.shape) == 2
        assert len(np.unique(tri)) <= 3

    def _one_hot(self, tri:np.ndarray, num_classes:int):
        h, w = tri.shape
        tri = np.eye(num_classes)[np.reshape(tri.astype(np.int32), (-1,))]
        one_hot = np.reshape(tri, (h, w, num_classes))
        return one_hot

    def _format(self, bgr, tri):
        h, w = tri.shape
        if h % 32 == 0 and w % 32 == 0:
            bgr = np.pad(bgr, ((32, 32), (32, 32), (0, 0)), mode="reflect")
            tri = np.pad(tri, ((32, 32), (32, 32)), mode="reflect")
        else:
            target_h = 32 * ((h - 1) // 32 + 1)
            target_w = 32 * ((w - 1) // 32 + 1)
            pad_h = target_h - h
            pad_w = target_w - w
            bgr = np.pad(bgr, ((32, pad_h + 32), (32, pad_w + 32), (0, 0)), mode="reflect")
            tri = np.pad(tri, ((32, pad_h + 32), (32, pad_w + 32)), mode="reflect")

        # mean & std
        mean = np.reshape(np.array([0.485, 0.456, 0.406]), (3, 1, 1))
        std = np.reshape(np.array([0.229, 0.224, 0.225]), (3, 1, 1))
        # swap color axis
        rgb = bgr[:, :, ::-1]
        rgb = rgb.transpose((2, 0, 1)).astype(np.float32)
        # normalize image
        rgb = (rgb / 255. - mean) / std
        tri = np.transpose(np.expand_dims(tri, axis=2), (2, 0, 1)).astype(np.float32)
        batch_rgb = rgb[None, ...]
        batch_tri = tri[None, ...]
        return batch_rgb, batch_tri

    def _forward(self, batch_rgb, batch_tri):
        alpha = self.engine.inference(batch_rgb, batch_tri)
        return alpha[0, 0, ...].astype(np.float32)

    def _post(self, alpha, tri, src_h, src_w):
        alpha = (alpha * 255).astype(np.uint8)
        alpha = alpha[32:src_h + 32, 32:src_w + 32]
        alpha[tri == 2] = 255
        alpha[tri == 0] = 0
        return alpha

    def initialize(self):
        self.engine.initialize()

    def pipeline(self, bgr, tri):
        self._assert(bgr, tri)
        h, w = tri.shape
        batch_rgb, batch_tri = self._format(bgr, tri)
        alpha = self._forward(batch_rgb, batch_tri)
        return self._post(alpha, tri, h, w)



if __name__ == '__main__':
    import cv2
    config = dict(
        type='torch',
        device='cuda:0',
        parameters='E:/experiment/fgi-matting/checkpoints/fgi-dim.pt'
    )
    lib = LibFGI(config={'engine':config})
    lib.initialize()
    path_img = 'data/matte/image/doll.png'
    path_tri = 'data/matte/trimap/doll.png'
    bgr = cv2.imread(path_img, cv2.IMREAD_COLOR)
    tri = cv2.imread(path_tri, cv2.IMREAD_GRAYSCALE)
    print(np.unique(tri))
    tri[tri == 128] = 1
    tri[tri == 255] = 2
    matte = lib.pipeline(bgr, tri)
    cv2.imshow('matte', matte)
    cv2.waitKey(0)