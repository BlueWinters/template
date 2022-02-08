
import cv2
import numpy as np
import logging
from core.engine import *


class LibSIM:
    """
    """
    TriMapSourceLabel = [0, 1, 2]
    TriMapTargetLabel = [0, 128, 255]

    """
    """
    def __init__(self, config):
        self.config = config
        self.engine_cls = create_engine(config['engine_cls'])
        self.engine_sim = create_engine(config['engine_sim'])
        # some common config
        self.stride = self.config['stride']
        self.num_classes = self.config['num_classes']

    def __del__(self):
        logging.warning('delete engine LibSIM')

    def _assert(self, bgr, tri):
        assert len(bgr.shape) == 3
        assert len(tri.shape) == 2
        assert len(np.unique(tri)) <= 3

    def _transform(self, image, scale=255.):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mean = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
        std = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)
        image_scale = image / scale
        image_trans = (image_scale - mean) / std
        # image_scale = torch.from_numpy(image_scale.transpose(2, 0, 1)).float()
        # image_trans = torch.from_numpy(image_trans.transpose(2, 0, 1)).float()
        return image_scale, image_trans

    def _trimap_to_2chn(self, trimap):
        h, w = trimap.shape[:2]
        trimap_2chn = np.zeros((h, w, 2), dtype=np.float32)
        trimap_2chn[:, :, 0] = (trimap == 0)
        trimap_2chn[:, :, 1] = (trimap == 255)
        return trimap_2chn

    def _trimap_to_clks(self, trimap, L=320):
        def dt(a):
            return cv2.distanceTransform((a * 255).astype(np.uint8), cv2.DIST_L2, 0)

        h, w = trimap.shape[:2]
        clicks = np.zeros((h, w, 6), dtype=np.float32)
        for k in range(2):
            if (np.count_nonzero(trimap[:, :, k]) > 0):
                dt_mask = -dt(1 - trimap[:, :, k]) ** 2
                clicks[:, :, 3 * k] = np.exp(dt_mask / (2 * ((0.02 * L) ** 2)))
                clicks[:, :, 3 * k + 1] = np.exp(dt_mask / (2 * ((0.08 * L) ** 2)))
                clicks[:, :, 3 * k + 2] = np.exp(dt_mask / (2 * ((0.16 * L) ** 2)))
        return clicks

    def _format(self, image, trimap):
        stride = self.stride
        h, w = image.shape[:2]
        pad_h = (h // stride + 1) * stride - h
        pad_w = (w // stride + 1) * stride - w
        pad_tri = ((0, pad_h), (0, pad_w))
        pad_img = ((0, pad_h), (0, pad_w), (0, 0))

        assert len(np.unique(trimap)) <= 3
        trimap[trimap == LibSIM.TriMapSourceLabel[0]] = LibSIM.TriMapTargetLabel[0]
        trimap[trimap == LibSIM.TriMapSourceLabel[1]] = LibSIM.TriMapTargetLabel[1]
        trimap[trimap == LibSIM.TriMapSourceLabel[2]] = LibSIM.TriMapTargetLabel[2]
        # trimap = cv2.copyMakeBorder(trimap, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        # image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        trimap = np.pad(trimap, pad_tri, mode='constant', constant_values=0)
        image = np.pad(image, pad_img, mode='reflect')
        image_scale, image_trans = self._transform(image, scale=255.)
        trimap_2chn = self._trimap_to_2chn(trimap)
        trimap_clks = self._trimap_to_clks(trimap_2chn, 320)

        image = np.expand_dims(image.transpose((2, 0, 1)), axis=0)
        trimap = np.expand_dims(trimap, axis=(0, 1))
        image_scale = np.expand_dims(image_scale.transpose((2, 0, 1)), axis=0)
        image_trans = np.expand_dims(image_trans.transpose((2, 0, 1)), axis=0)
        trimap_2chn = np.expand_dims(trimap_2chn.transpose((2, 0, 1)), axis=0)
        trimap_clks = np.expand_dims(trimap_clks.transpose((2, 0, 1)), axis=0)

        inputs = {
            'image': image.astype(np.float32),
            'trimap': trimap.astype(np.float32),
            'image_scale': image_scale.astype(np.float32),
            'image_trans': image_trans.astype(np.float32),
            'trimap_2chn': trimap_2chn.astype(np.float32),
            'trimap_clks': trimap_clks.astype(np.float32),
            'origin_h': h,
            'origin_w': w,
        }
        return inputs

    def _inference_cls(self, image, trimap, return_cam=False):
        N, C, H, W = image.shape
        cls_input = np.concatenate([image, trimap / 255.], axis=1)
        output, cam = self.engine_cls.inference(cls_input)
        cam = cv2.resize(np.transpose(cam[0, ...], (1, 2, 0)), (W, H), interpolation=cv2.INTER_LINEAR)
        batch_cam = np.expand_dims(np.transpose(cam, (2, 0, 1)), axis=0)
        if return_cam: return batch_cam, output

    def _extract_semantic_trimap(self, batch_inputs):
        image, trimap = batch_inputs['image'], batch_inputs['trimap']
        N, C, H, W = image.shape
        cam = np.zeros((N, self.num_classes, H, W), dtype=np.float32)
        weight = np.zeros((N, self.num_classes, H, W), dtype=np.float32)
        for step in [320, 800]:
            xs = list(range(0, W - step, step // 2)) + [W - step]
            ys = list(range(0, H - step, step // 2)) + [H - step]
            for i in ys:
                for j in xs:
                    patch_tri = trimap[:, :, i:i + step, j:j + step]
                    if (patch_tri == 128).sum() == 0: continue
                    patch_img = image[:, :, i:i + step, j:j + step]
                    patch_cam, out = self._inference_cls(patch_img, patch_tri, return_cam=True)
                    cam[:, :, i:i + step, j:j + step] += patch_cam
                    weight[:, :, i:i + step, j:j + step] += 1
        cam = cam / np.clip(weight, a_min=1, a_max=None)
        cam_norm = (cam - cam.min()) / (cam.max() - cam.min())
        return cam_norm * np.where(trimap == 128, 1, 0).astype(np.float32)

    def _post(self, batch_inputs, alpha):
        oh = batch_inputs['origin_h']
        ow = batch_inputs['origin_w']
        trimap = batch_inputs['trimap']
        alpha = np.clip(alpha, a_min=0., a_max=1.)
        alpha[trimap == 0] = 0
        alpha[trimap == 255] = 1
        alpha = (alpha[:, :, 0:oh, 0:ow] * 255).astype(np.uint8)
        return alpha[0, 0, ...]

    def _extract_alpha(self, batch_inputs, semantic_trimap):
        image_scale = batch_inputs['image_scale']
        trimap_2chn = batch_inputs['trimap_2chn']
        image_trans = batch_inputs['image_trans']
        trimap_clks = batch_inputs['trimap_clks']
        alpha = self.engine_sim.inference(image_scale, trimap_2chn, image_trans, trimap_clks, semantic_trimap)
        return alpha

    def initialize(self):
        self.engine_cls.initialize()
        self.engine_sim.initialize()

    def pipeline(self, bgr, tri):
        self._assert(bgr, tri)
        batch_inputs_dict = self._format(bgr, tri.copy())
        batch_semantic_trimap = self._extract_semantic_trimap(batch_inputs_dict)
        batch_alpha = self._extract_alpha(batch_inputs_dict, batch_semantic_trimap)
        return self._post(batch_inputs_dict, batch_alpha)




if __name__ == '__main__':
    import cv2
    config = dict(
        num_classes=20,
        stride=8,
        engine_cls=dict(
            type='torch',
            device = 'cuda:0',
            parameters = 'E:/experiment/semantic-image-matting/checkpoints/export/sim.classifier.pt',
        ),
        engine_sim=dict(
            type='torch',
            device='cuda:0',
            parameters='E:/experiment/semantic-image-matting/checkpoints/export/sim.model.pt',
        ),
    )
    lib = LibSIM(config=config)
    lib.initialize()
    path_img = 'data/matte/image/doll.png'
    path_tri = 'data/matte/trimap/doll.png'
    bgr = cv2.imread(path_img, cv2.IMREAD_COLOR)
    tri = cv2.imread(path_tri, cv2.IMREAD_GRAYSCALE)
    print(np.unique(tri))
    tri[tri == 128] = 1
    tri[tri == 255] = 2
    print(np.unique(tri))
    matte = lib.pipeline(bgr, tri)
    cv2.imshow('matte', matte)
    cv2.waitKey(0)