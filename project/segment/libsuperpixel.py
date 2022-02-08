
import os
import cv2
import math
import numpy as np
from skimage.segmentation import slic, mark_boundaries


class LibSuperPixel:
    default_mesh_size = 8

    def __init__(self):
        pass

    @staticmethod
    def super_pixel(bgr, *args, **kwargs):
        mesh_size = kwargs['mesh_size'] if 'mesh_size' in kwargs \
            else LibSuperPixel.default_mesh_size
        div = bgr.size / (mesh_size * mesh_size)
        n_segments = math.ceil(div)
        segments = slic(bgr, n_segments=n_segments, compactness=10, start_label=1)
        return segments, n_segments

    @staticmethod
    def visual_boundaries(bgr, seg):
        out = mark_boundaries(bgr, seg)
        return out, seg

    @staticmethod
    def countmax_value(array:np.ndarray):
        count = np.bincount(array.astype(np.int32))
        value = np.argmax(count)
        return value

    @staticmethod
    def merge_segment(self, super_pixels, seg, mat):
        labels = np.unique(super_pixels)
        mat_bin = np.where(mat > 127, 1, 0)
        new_seg = np.zeros_like(seg)
        new_mat = np.zeros_like(mat)
        for n in labels:
            # for seg
            seg_part = seg[super_pixels == n, ...]
            index = self.countmax_value(seg_part)
            new_seg[super_pixels == n, ...] = index
            # for mat
            mat_bin_part = mat_bin[super_pixels == n, ...]
            index = self.countmax_value(mat_bin_part)
            assert index == 0 or index == 1
            mat_part = (mat[(super_pixels == n) & (mat_bin == index), ...])
            # if len(mat_part) != len(mat_bin_part):
            #     print('go: {}'.format(np.mean(mat_part)))
            new_mat[super_pixels == n, ...] = np.mean(mat_part)
        return new_mat, new_seg

    @staticmethod
    def merge(bgr, seg, mat, *args, **kwargs):
        super_pixels, _ = LibSuperPixel.super_pixel(bgr)
        new_mat, new_seg = LibSuperPixel.merge_segment(super_pixels, seg, mat)
        return new_mat, new_seg

    @staticmethod
    def split(bgr, *args, **kwargs):
        super_pixels, _ = LibSuperPixel.super_pixel(bgr)
        new_bgr, _ = LibSuperPixel.visual_boundaries(bgr, super_pixels)
        new_bgr = np.round(new_bgr*255).astype(np.uint8)
        return new_bgr