
import os
import cv2
import numpy as np
import tempfile
from moviepy.editor import VideoFileClip
from typing import Union, List, Tuple, Callable, Any


class LibVideoReader:
    """
    """
    @staticmethod
    def _decode_codec(raw_codec_format:int):
        decoded_codec_format = (chr((raw_codec_format & 0xFF)),
                                chr((raw_codec_format & 0xFF00) >> 8),
                                chr((raw_codec_format & 0xFF0000) >> 16),
                                chr((raw_codec_format & 0xFF000000) >> 24))
        return decoded_codec_format

    """
    """
    def __init__(self, video:Union[str,bytes]):
        self._config(self._open(video))

    def _open(self, video) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(cv2.CAP_FFMPEG)
        if isinstance(video, bytes):
            stream = video
            file = tempfile.TemporaryFile()
            file.dump(stream)
            path = file.name
        else:
            path = video  # str is for both file path
        capture.open(path)
        return capture

    def _config(self, capture:cv2.VideoCapture):
        if capture.isOpened():
            self.capture = capture
            self.w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(capture.get(cv2.CAP_PROP_FPS))
            self.fourcc = LibVideoReader._decode_codec(int(capture.get(cv2.CAP_PROP_FOURCC)))
            self.num_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.num_sec = int(self.num_frame / self.fps)

    def desc(self):
        return dict(
            w=self.w, h=self.h, fps=self.fps,
            num_frames=self.num_frame,
            fourcc=self.fourcc,
            num_sec=self.num_sec
        )

    def __iter__(self):
        return self

    def __next__(self):
        ret, bgr = self.capture.read()
        if ret is False:
            raise StopIteration
        return bgr

    def is_open(self) -> bool:
        if hasattr(self, 'capture'):
            return self.capture.isOpened()
        return False

    def read(self) -> Tuple[Any,np.ndarray]:
        ret, bgr = self.capture.read()
        return ret, bgr

    """
    read all frames by step
    """
    def sequential(self, step:int=1) -> List[np.ndarray]:
        assert 0 < step < self.num_frame
        data = list()
        for n in range(self.num_frame):
            bgr = self.read()[1]
            if n % step == 0:
                data.append(bgr)
        return data

    def reset_position_index(self, index:int):
        self.cur = min(max(index, 0), self.num_frame - 1)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.cur)

    def reset_position_ratio(self, ratio:float):
        self.cur = int(ratio * self.num_frame + 0.5)
        self.cur = min(max(self.cur, 0), self.num_frame - 1)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.cur)

    def sample(self, step:int, path:str):
        assert self.is_open()
        assert 0 < step < self.num_frame
        for n in range(self.num_frame):
            bgr = self.read()[1]
            if n % step == 0:
                save_path = '{}/{:05d}.png'.format(path, n)
                cv2.imencode('.png', bgr)[1].tofile(save_path)
        print('finish sample...')



class LibVideoWriter:
    """
    """
    @staticmethod
    def _default_fourcc():
        return ('X', 'V', 'I', 'D')

    """
    """
    def __init__(self, config:dict):
        self._config(config)

    def _config(self, config:dict):
        self.fps = config['fps']
        self.w = config['w']
        self.h = config['h']
        self.fourcc = config['fourcc'] if 'fourcc' in config \
            else LibVideoWriter._default_fourcc()
        assert len(self.fourcc) == 4

    def _writer(self, path:str):
        def _open_writer(_fourcc:Tuple):
            writer = cv2.VideoWriter()
            code = cv2.VideoWriter_fourcc(*_fourcc)
            writer.open(path, code, self.fps, (self.w, self.h), True)
            return writer.isOpened(), writer

        is_open, video_writer = _open_writer(self.fourcc)
        if is_open is False:
            video_writer.release()
            print('reset to default fourcc')
            return _open_writer(self._default_fourcc())[1]
        return video_writer

    """
    for single step writer
    usage for local file:
        writer.dump(frame_list, path_out)
    usage for bytes:
        stream = writer.dump(frame_list, None)
    """
    def _serialize(self, seq:List[np.ndarray], writer:cv2.VideoWriter):
        for n, frame in enumerate(seq):
            writer.write(frame)
        writer.release()

    def _dump_as_file(self, path:str, seq:List[np.ndarray]):
        assert os.path.exists(os.path.split(path)[0])
        self._serialize(seq, writer=self._writer(path))
        return path

    def _dump_as_bytes(self, seq:List[np.ndarray]):
        file = tempfile.TemporaryFile(suffix='.avi')
        self._serialize(seq, writer=self._writer(file.name))
        file.seek(0)
        return bytes(file.read())

    def dump(self, seq:List[np.ndarray], path:Union[str,None]) -> Union[str,bytes]:
        return self._dump_as_bytes(seq) if path is None \
            else self._dump_as_file(path, seq)

    """
    for multi step writer
    usage:
        handle = writer.open(path_out)
        for frame in frame_list:
            handle.send(frame)
    """
    def open(self, path:str):
        assert os.path.exists(os.path.split(path)[0])
        writer = self._writer(path)
        g = self._serialize_yield(writer)
        g.__next__()
        return g

    def _serialize_yield(self, writer:cv2.VideoWriter):
        counter = 0
        while True:
            counter += 1
            img = yield self
            writer.write(img)

    """
    merge some images into one video
    """
    def merge_from(self, path_image:str, path_video:str):
        assert os.path.exists(path_image)
        seq = list()
        for name in os.listdir(path_image):
            path = '{}/{}'.format(path_image, name)
            bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            seq.append(bgr)
        self.dump(seq, path_video)

    """
    """
    @staticmethod
    def concatenate(path_in_list, path_out, axis:int=0):
        assert axis == 0 or axis == 1
        reader_list = list()
        config = None
        for path_in in path_in_list:
            reader = LibVideoReader(path_in)
            assert reader.is_open()
            config = reader.desc() if config is None else config
            assert config == reader.desc()
            reader_list.append(reader)
        axis_str = 'h' if axis == 0 else 'w'
        config[axis_str] *= len(reader_list)
        writer = LibVideoWriter(config)
        buffer = writer.open(path_out)
        for frames in zip(*reader_list):
            image = np.concatenate(frames, axis=axis)
            buffer.send(np.ascontiguousarray(image))
        print('finish...')



class LibVideoProcess:
    """
    """
    class Counter:
        def __init__(self, max_value):
            self.index = 0
            self.max = max_value

        def __call__(self, *args, **kwargs):
            cur = self.index
            self.index = min(self.index+1, self.max)
            return cur

    """
    """
    def __init__(self, path_in:str):
        assert os.path.exists(path_in)
        self.handle = VideoFileClip(path_in)

    def process(self, path_ot:str, step_fcn:Callable=None):
        step_fcn = step_fcn if step_fcn is not None else lambda gf,t: gf(t)
        self.handle.fl(step_fcn).write_videofile(path_ot, logger=None)

    def replace(self, path_ot:str, bgr_list:List[np.ndarray]):
        counter = LibVideoProcess.Counter(len(bgr_list)-1)
        step = lambda gf,t: bgr_list[counter()][:,:,::-1]
        self.handle.fl(step).write_videofile(path_ot, logger=None)



if __name__ == '__main__':
    pass
