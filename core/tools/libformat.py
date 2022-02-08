
import cv2
import numpy as np
from typing import Dict, Tuple, List, Callable


class LibImageFormat:
    def __init__(self):
        self.param = dict()

    @staticmethod
    def unfold_parameter(param:Dict, key_seq:Tuple[str,...], **kwargs):
        data = [param[key] if key not in kwargs else kwargs[key] for key in key_seq]
        return data

    @staticmethod
    def forward(*args, **kwargs):
        ...

    @staticmethod
    def backward(*args, **kwargs):
        ...



class Resize(LibImageFormat):
    """
    """
    Method = dict(
        nearest=cv2.INTER_NEAREST,
        linear=cv2.INTER_LINEAR,
        cubic=cv2.INTER_CUBIC,
    )

    """
    """
    def __init__(self, ):
        super(Resize, self).__init__()

    @staticmethod
    def calculate_size(src_h:int, src_w:int, dst_h:[int,None], dst_w:[int,None], *args, **kwargs):
        assert not (dst_h is None and dst_w is None)
        if dst_h is not None and dst_w is not None:
            return dst_h, dst_w
        align = dict(axis=1, size=dst_w) if dst_h is None else dict(axis=0, size=dst_h)
        return Resize.align_by_ratio(src_h, src_w, **align)

    @staticmethod
    def align_by_ratio(src_h:int, src_w:int, axis:int, size:int=None):
        src_ratio = float(src_h / src_w)
        if axis == 0:
            dst_h = src_h if size is None else size
            dst_w = int(dst_h / src_ratio + 0.5)
        else:
            dst_w = src_w if size is None else size
            dst_h = int(dst_w * src_ratio + 0.5)
        return dst_h, dst_w

    @staticmethod
    def forward(im:np.ndarray, dst_h:[int,None], dst_w:[int,None], method:[str,None]='linear'):
        src_h, src_w = im.shape[:2]
        dst_h, dst_w = Resize.calculate_size(src_h, src_w, dst_h, dst_w)
        interpolation = Resize.Method['linear' if method is None else method]
        resized = cv2.resize(im, (dst_w, dst_h), interpolation=interpolation)
        param_dict = dict(src_h=src_h, src_w=src_w, dst_h=dst_h, dst_w=dst_w, method=method)
        return resized, param_dict

    @staticmethod
    def backward(im:np.ndarray, param:Dict, **kwargs):
        cur_h, cur_w = im.shape[:2]
        src_h, src_w, method = LibImageFormat.unfold_parameter(
            param, ('src_h', 'src_w', 'method'), **kwargs)
        interpolation = Resize.Method['linear' if method is None else method]
        resized = cv2.resize(im, (src_w, src_h), interpolation=interpolation)
        param_dict = dict(src_h=src_h, src_w=src_w, dst_h=cur_h, dst_w=cur_w, method=method)
        return resized, param_dict

    @staticmethod
    def benchmark():
        bgr = cv2.imread('data/dog.png')
        bgr_f, param = Resize.forward(bgr, dst_h=512, dst_w=None, method='linear')
        bgr_b, _ = Resize.backward(bgr_f, param)
        print(bgr.shape, bgr_f.shape, bgr_b.shape)
        cv2.imshow('bgr', bgr)
        cv2.imshow('bgr_f', bgr_f)
        cv2.imshow('bgr_b', bgr_b)
        cv2.waitKey(0)



class ResizeAlign(Resize):
    def __init__(self):
        super(ResizeAlign, self).__init__()

    @staticmethod
    def calculate_size(axis:str, src_h:int, src_w:int, size, *args, **kwargs):
        assert axis == 'long' or axis == 'short'
        if axis == 'long':
            axis = 0 if src_h >= src_w else 1
            return Resize.align_by_ratio(src_h, src_w, axis=axis, size=size)
        else:
            axis = 0 if src_h <= src_w else 1
            return Resize.align_by_ratio(src_h, src_w, axis=axis, size=size)

    @staticmethod
    def forward(im:np.ndarray, axis:str, size:[int,None], method:str='linear'):
        src_h, src_w = im.shape[:2]
        dst_h, dst_w = ResizeAlign.calculate_size(axis, src_h, src_w, size)
        interpolation = ResizeAlign.Method['linear' if method is None else method]
        resized = cv2.resize(im, (dst_w, dst_h), interpolation=interpolation)
        param_dict = dict(src_h=src_h, src_w=src_w,
                          dst_h=dst_h, dst_w=dst_w, axis=axis, size=size, method=method)
        return resized, param_dict

    @staticmethod
    def backward(im:np.ndarray, param:Dict, **kwargs):
        cur_h, cur_w = im.shape[:2]
        src_h, src_w, method = LibImageFormat.unfold_parameter(
            param, ('src_h', 'src_w', 'method'), **kwargs)
        interpolation = ResizeAlign.Method['linear' if method is None else method]
        resized = cv2.resize(im, (src_w, src_h), interpolation=interpolation)
        param_dict = dict(src_h=src_h, src_w=src_w,
                          dst_h=cur_h, dst_w=cur_w, method=method)
        return resized, param_dict

    @staticmethod
    def benchmark():
        bgr = cv2.imread('data/dog.png')
        bgr_f, param = ResizeAlign.forward(bgr, axis='long', size=256)
        bgr_b, _ = ResizeAlign.backward(bgr_f, param)
        print(bgr.shape, bgr_f.shape, bgr_b.shape)
        cv2.imshow('bgr', bgr)
        cv2.imshow('bgr_f', bgr_f)
        cv2.imshow('bgr_b', bgr_b)
        cv2.waitKey(0)


class Padding(LibImageFormat):
    def __init__(self):
        super(Padding, self).__init__()

    @staticmethod
    def forward(im:np.ndarray, dst_h:int, dst_w:int, padding_value:int=0):
        src_h, src_w = im.shape[:2]
        assert src_h <= dst_h and src_w <= dst_w
        assert 0 <= padding_value <= 255
        padding_h = dst_h - src_h
        padding_w = dst_w - src_w
        top = padding_h // 2
        bot = padding_h - padding_h // 2
        lft = padding_w // 2
        rig = padding_w - padding_w // 2
        padding_size = ((top, bot), (lft, rig)) if len(im.shape) == 2 \
            else ((top, bot), (lft, rig), (0, 0))
        padding_im = np.pad(im, padding_size,
            mode='constant', constant_values=padding_value)
        param_dict = dict(top=top, bot=bot, lft=lft, rig=rig)
        return padding_im, param_dict

    @staticmethod
    def backward(im:np.ndarray, param:Dict, **kwargs):
        cur_h, cur_w = im.shape[:2]
        top, bot, lft, rig = LibImageFormat.unfold_parameter(
            param, ('top', 'bot', 'lft', 'rig'), **kwargs)
        part = im[top:cur_h-bot, lft:cur_w-rig, ...]
        param_dict = dict(top=top, bot=bot, lft=lft, rig=rig)
        return part, param_dict



class Split(LibImageFormat):
    def __init__(self):
        super(LibImageFormat, self).__init__()

    @staticmethod
    def forward(im:np.ndarray, max_ratio:float, axis:int):
        src_h, src_w = im.shape[:2]
        ratio = src_h / src_w if axis == 0 else src_w / src_h
        if ratio > max_ratio:
            data = list()
            if axis == 0:
                size_h = int(src_w*max_ratio+0.5)
                num = int(np.ceil(src_h/size_h))
                for n in range(num):
                    beg = n * size_h
                    end = (n+1) * size_h if not (n == num-1) else src_h
                    data.append(im[beg:end, :, ...])
                param_dict = dict(src_h=src_h, src_w=src_w, axis=axis)
            else:
                size_w = int(src_h * max_ratio + 0.5)
                num = int(np.ceil(src_w / size_w))
                for n in range(num):
                    beg = n * size_w
                    end = (n+1) * size_w if not (n == num-1) else src_w
                    data.append(im[:, beg:end, ...])
                param_dict = dict(src_h=src_h, src_w=src_w, axis=axis)
            return data, param_dict
        return [im], dict(src_h=src_h, src_w=src_w, axis=axis)

    @staticmethod
    def backward(im_list:List[np.ndarray], param:Dict, **kwargs):
        src_h, src_w, axis = LibImageFormat.unfold_parameter(
            param, ('src_h', 'src_w', 'axis'), **kwargs)
        cat = np.concatenate(im_list, axis=axis)
        param_dict = dict(src_h=src_h, src_w=src_w, axis=axis)
        return cat, param_dict

    @staticmethod
    def benchmark():
        bgr = cv2.imread('data/dog.png')
        bgr_list, param = Split.forward(bgr, max_ratio=0.5, axis=0)
        bgr_b, _ = Split.backward(bgr_list, param)
        print(bgr.shape, [bgr_f.shape for bgr_f in bgr_list], bgr_b.shape)
        assert bgr.shape == bgr_b.shape
        cv2.imshow('bgr', bgr)
        cv2.imshow('bgr_b', bgr_b)
        cv2.waitKey(0)




class LibFormatSequential:
    """
    """
    @staticmethod
    def sequential(*args, **kwargs):
        pass

    def __init__(self):
        pass

    @staticmethod
    def forward(im:np.ndarray, format_list:List):
        assert len(format_list) > 0
        param_list = list()
        for format, config in format_list:
            assert issubclass(format, LibImageFormat)
            im, param = format.forward(im, **config)
            param['type'] = format
            param_list.append(param)
        return im, param_list

    @staticmethod
    def backward(image:np.ndarray, param_list:List[Dict], **kwargs):
        param_dict_list = list()
        for param in param_list:
            class_type = param['type']
            assert issubclass(class_type, LibImageFormat)
            result = class_type.backward(image, param)
            image = result[0]
            param_dict_list.append(result[1])
        return image, param_dict_list

    @staticmethod
    def benchmark():
        bgr = cv2.imread('data/dog.png')
        format_list = [
            [ResizeAlign, dict(axis='long', size=256, method='linear')],
            [Padding, dict(dst_h=256, dst_w=256, padding_value=0)],
        ]
        image_f, param_list = LibFormatSequential.forward(bgr, format_list)
        image_b, _ = LibFormatSequential.backward(bgr, param_list)
        print(image_f.shape, image_b.shape)
        cv2.imshow('bgr', bgr)
        cv2.imshow('bgr_f', image_f)
        cv2.imshow('bgr_b', image_b)
        cv2.waitKey(0)




if __name__ == '__main__':
    # Resize.benchmark()
    # ResizeAlign.benchmark()
    # Split.benchmark()
    LibFormatSequential.benchmark()