
import cv2
import os
import numpy as np
import tempfile
from typing import Union, List


class ContainerImage:
    def __init__(self, path:str):
        self.path = path
        self.rgb = ContainerImage.read(path)

    @staticmethod
    def concatenate(images_list:List):
        N = len(images_list)
        h, w = images_list[0].shape[:2]
        data = list()
        for image in images_list:
            assert (h, w) == image.shape[:2]
            rgb = image if len(image.shape) == 3 \
                else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            data.append(rgb)
        nh, nw = N * h, N * w
        axis = 0 if abs(float(nh / w) - 1.) < abs(float(h / nw) - 1.) else 1
        return np.concatenate(data, axis=axis).copy()

    @staticmethod
    def read(path):
        assert os.path.exists(path)
        bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert bgr is not None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @property
    def bgr(self):
        return cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR)






class ContainerVideo:
    def __init__(self, video:Union[str, bytes]):
        self._config(*self._open(video))

    def _open(self, video):
        # if hasattr(se)
        capture = cv2.VideoCapture(cv2.CAP_FFMPEG)
        if isinstance(video, bytes):
            stream = video
            file = tempfile.TemporaryFile()
            file.dump(stream)
            path = file.name
            extension = ''
        else:
            path = video  # str is for both file path
            extension = os.path.splitext(video)[1]
        capture.open(path)
        return capture, extension

    def _config(self, capture, extension:str):
        if capture.isOpened():
            self.ext = extension
            self.capture = capture
            self.w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(capture.get(cv2.CAP_PROP_FPS))
            self.fourcc = ContainerVideo._decode_codec(int(capture.get(cv2.CAP_PROP_FOURCC)))
            self.num_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.num_msec = int(self.num_frame / self.fps)
            # some variables
            self.cur = 0
            self.index_beg = None
            self.index_end = None
            self.marks = None

    @staticmethod
    def _decode_codec(raw_codec_format:int):
        decoded_codec_format = (chr((raw_codec_format & 0xFF)),
                                chr((raw_codec_format & 0xFF00) >> 8),
                                chr((raw_codec_format & 0xFF0000) >> 16),
                                chr((raw_codec_format & 0xFF000000) >> 24))
        return decoded_codec_format

    def format_summary(self):
        text = 'height: {}\nwidth: {}\nfps: {}\nframes: {}\ntime: {}s'.format(
            self.h, self.w, self.fps, self.num_frame, self.num_msec)
        return text

    def format_time(self):
        cur_msec = int(self.capture.get(cv2.CAP_PROP_POS_MSEC))
        num_msec = self.num_msec
        all_H = num_msec // (3600 * 1000)
        all_M = num_msec % (3600 * 1000) // (60 * 1000)
        all_S = num_msec // 1000
        cur_H = cur_msec // (3600 * 1000)
        cur_M = cur_msec % (3600 * 1000) // (60 * 1000)
        cur_S = cur_msec // 1000
        return '{:02d}:{:02d}:{:02d} / {:02d}:{:02d}:{:02d}'.format(cur_H, cur_M, cur_S, all_H, all_M, all_S)

    def format_frame(self):
        return '{:d} / {:d}'.format(self.cur, self.num_frame)

    def reset_position_index(self, index:int):
        self.cur = min(max(index, 0), self.num_frame - 1)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.cur)

    def reset_position_ratio(self, ratio:float):
        self.cur = int(ratio * self.num_frame + 0.5)
        self.cur = min(max(self.cur, 0), self.num_frame - 1)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.cur)

    def forward(self, num:int):
        self.cur = min(self.cur + num, self.num_frame-1)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.cur)

    def backward(self, num:int):
        self.cur = max(self.cur - num, 0)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.cur)

    def read(self):
        self.cur += 1
        return self.capture.read()

    def current_index(self):
        return self.cur

    def current_ratio(self):
        return int(float(self.cur) / self.num_frame * 100.)

    def set_begin(self):
        self.index_beg = self.cur

    def set_end(self):
        self.index_end = self.cur

    def set_whole(self):
        self.index_beg = 0
        self.index_end = self.num_frame - 1

    def set_marks(self, marks):
        self.marks = marks

    def get_marks(self):
        if hasattr(self, 'marks'):
            return self.marks
        return None

    def valid(self):
        return bool(self.index_beg is not None) & \
               bool(self.index_end is not None) & \
               bool(self.marks is not None)

