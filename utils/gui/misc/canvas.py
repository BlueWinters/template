
import cv2
import numpy as np
from typing import List, Tuple, Union


class Canvas:
    def __init__(self, *args, **kwargs):
        # other fcn
        self.quantify = lambda v: int(v + 0.5)
        # image resource
        self.image_src = kwargs['rgb'].copy()
        self.image_vis = self._resize(self.image_src, 1.)
        # current display ratio
        self.ratio = self._init_ratio(size=kwargs['size'])  # h,w
        # for all types methods
        self.handle_dict = Canvas._handle_dict()
        # current method
        self.coating_list = list()

    @staticmethod
    def _handle_dict():
        return MarkClassDict(
            [Mark.name(), Rectangle.name(), Stroke.name(), Polygon.name()],
            [Mark, Rectangle, Stroke, Polygon]
        )

    @property
    def _handle_object(self):
        return self.coating_list[-1]

    def _init_ratio(self, size):
        fh, fw = size
        ih, iw = self.image_src.shape[:2]
        fR = float(fh / fw)
        iR = float(ih / iw)
        if fR > iR:
            base_size = fw
            self.config_max_ratio = 10.
            self.config_min_ratio = float((base_size - 2) / iw)
        else:
            base_size = fh
            self.config_max_ratio = 10.
            self.config_min_ratio = float((base_size - 2) / ih)
        return self.config_min_ratio

    def _resize(self, rgb:np.ndarray, ratio:float) -> np.ndarray:
        # rgb = self.image_src
        src_h, src_w, src_c = rgb.shape
        dst_h = self.quantify(src_h * ratio)
        dst_w = self.quantify(src_w * ratio)
        if rgb.shape[:2] != (dst_h, dst_w):
            resized = cv2.resize(rgb, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
            return resized
        return rgb

    def _canvas_shape(self):
        return self.image_src.shape[:2]

    def __len__(self):
        return len(self.coating_list)

    def mouse_press_event(self, *args):
        x, y, event = args
        self._handle_object.mouse_press_event(x, y, event, self.ratio)
        image = self._handle_object.mouse_press_draw(self.image_src.copy(), 1., event)
        return self._resize(image, self.ratio)

    def mouse_move_event(self, *args):
        x, y, event = args
        self._handle_object.mouse_move_event(x, y, event, self.ratio)
        image = self._handle_object.mouse_move_draw(self.image_src.copy(), 1., event)
        return self._resize(image, self.ratio)

    def mouse_release_event(self, *args):
        x, y, event = args
        self._handle_object.mouse_release_event(x, y, event, self.ratio)
        image = self._handle_object.mouse_release_draw(self.image_src.copy(), 1., event)
        return self._resize(image, self.ratio)

    def mouse_key_press_event(self, *args):
        event, = args
        self._handle_object.mouse_key_press_event(event, self.ratio)
        image = self._handle_object.mouse_key_press_draw(self.image_src.copy(), 1., event)
        return self._resize(image, self.ratio)

    def wheel_event(self, *args):
        # change ratio
        sign, = args
        ratio = 0.02 * sign + self.ratio
        return self.set_scale(scale=ratio, update=True)

    def input_once_event(self, *args, **kwargs):
        if self._handle_object.input_once_event(*args, **kwargs):
            image = self._handle_object.input_once_draw(self.image_src.copy(), 1.)
            return self._resize(image, self.ratio)
        return self._resize(self.image_src.copy(), self.ratio)

    def update(self):
        image = self._handle_object.redraw_on(self.image_src.copy(), 1.)
        return self._resize(image, self.ratio)

    def change_handle(self, type:str):
        if len(self.coating_list) > 0:
            # if self.coating_list[-1].valid() is False:
            self.coating_list.pop(-1)
        self.coating_list.append(self.handle_dict[type](size=self._canvas_shape()))

    def get_results(self, **kwargs):
        data = self.handle_dict.string_dict(obj=list())
        for coating in self.coating_list:
            h, w, c = self.image_src.shape
            # if len(coating) > 0:
            data[str(coating)].extend(coating.get_results(h, w, **kwargs))
        return data

    def set_parameters(self, **kwargs):
        if isinstance(self._handle_object, Stroke) and 'radius' in kwargs:
            self._handle_object.radius = kwargs['radius']
        if isinstance(self._handle_object, Rectangle) and 'thick' in kwargs:
            self._handle_object.thick = kwargs['thick']

    def clear_marks(self):
        type = str(self.coating_list[-1])
        self.coating_list.clear()
        # self.image_vis = self._resize(self.image_src, 1.)
        return type

    def reset_image(self, rgb:np.ndarray):
        self.image_src = rgb.copy()
        self.image_vis = self._resize(self.image_src, 1.)
        for coating in self.coating_list:
            if isinstance(coating, Mark):
                coating.redraw_on(self.image_vis, 1.)
        return self._resize(self.image_vis, self.ratio)

    def set_scale(self, scale:float, update:bool=True) -> Union[np.ndarray, None]:
        # change ratio
        self.ratio = min(max(scale, self.config_min_ratio), self.config_max_ratio)
        # re-draw on canvas
        if update is True:
            return self.update()

    def get_scale(self):
        return self.ratio

    def set_color(self, color:Tuple[int,int,int]):
        # r, g, b = color
        self.coating_list[-1].set_color(color)

    def get_mark(self):
        return self._handle_object.name()




class Mark:
    # static variable
    __Name__ = 'mark'

    def __init__(self, *args, **kwargs):
        self.size = kwargs['size']
        self.history = list()
        self.quantify = lambda v: int(v+0.5)
        self.color = (255, 0, 0)  # rgb

    def __len__(self):
        return len(self.history)

    @staticmethod
    def name():
        return Mark.__Name__

    def mouse_press_event(self, *args):
        ...

    def mouse_move_event(self, *args):
        ...

    def mouse_release_event(self, *args):
        ...

    def mouse_key_press_event(self, *args):
        ...

    def wheel_event(self, *args):
        ...

    def input_once_event(self, *args, **kwargs):
        ...

    def mouse_press_draw(self, *args):
        ...

    def mouse_move_draw(self, *args):
        ...

    def mouse_release_draw(self, *args):
        ...

    def mouse_key_press_draw(self, *args):
        ...

    def input_once_draw(self, *args):
        ...

    def redraw_on(self, *args):
        ...

    def get_results(self, *args, **kwargs):
        ...

    def valid(self) -> bool:
        ...

    def set_color(self, color):
        ...



class Stroke(Mark):
    # static variable
    __Name__ = 'stroke'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.radius = 16.
        self.traces = list()
        self.data = None
        self.board = np.zeros(self.size, dtype=np.uint8)

    def __str__(self):
        return Stroke.__Name__

    @staticmethod
    def name():
        return Stroke.__Name__

    def _map_color(self, color):
        if tuple(color) == (255, 0, 0):
            return (2, 2, 2)
        if tuple(color) == (0, 0, 255):
            return (1, 1, 1)
        raise NotImplementedError

    def _draw_on_image(self, *args):
        if len(args) == 7:
            # draw with xyr
            rgb, x, y, radius, ratio, color, board = args
            xx = self.quantify(x * ratio)
            yy = self.quantify(y * ratio)
            rr = self.quantify(radius * ratio)
            cv2.circle(board, (xx, yy), rr, color=self._map_color(color), thickness=-1)
            mask = np.ones_like(rgb)
            mask[board == 2, 1:] = 0
            mask[board == 1, :2] = 0
            rgb *= mask
        if len(args) == 3:
            # draw with mask
            rgb, board, ratio = args
            mask = np.ones_like(rgb)
            mask[board == 2, 1:] = 0
            mask[board == 1, :2] = 0
            rgb *= mask

    def set_ratio(self, ratio):
        self.ratio = ratio

    def mouse_press_event(self, *args):
        x, y, event, ratio = args
        if 'left-button' in event:
            self.data = (x / ratio, y / ratio, self.radius / ratio, self.color)

    def mouse_move_event(self, *args):
        x, y, event, ratio = args
        if 'left-button' in event:
            self.data = (x / ratio, y / ratio, self.radius / ratio, self.color)

    def mouse_release_event(self, *args):
        x, y, event, ratio = args
        if 'left-button' in event:
            self.data = None  #(x / ratio, y / ratio, self.radius / ratio, self.color)

    def wheel_event(self, *args):
        pass

    def input_once_event(self, *args, **kwargs):
        mask, mode = kwargs['mask'], kwargs['mode']
        if len(mask.shape) == 2:
            # self.board[mask > 0] = self._map_color(self.color)[0]
            if mode == 'overwrite':
                self.board = mask
            if mode == 'select':
                self.board[mask > 0] = mask[mask > 0]
            return True
        return None

    def mouse_press_draw(self, *args):
        rgb, ratio, event = args
        if 'left-button' in event:
            if self.data != None:
                x, y, radius, color = self.data
                self._draw_on_image(rgb, x, y, radius, ratio, color, self.board)
        if 'right-button' in event:
            self._draw_on_image(rgb, self.board, ratio)
        return rgb

    def mouse_move_draw(self, *args):
        rgb, ratio, event = args
        if 'left-button' in event:
            if self.data != None:
                x, y, radius, color = self.data
                self._draw_on_image(rgb, x, y, radius, ratio, color, self.board)
        if 'right-button' in event:
            self._draw_on_image(rgb, self.board, ratio)
        return rgb

    def mouse_release_draw(self, *args):
        rgb, ratio, event = args
        self._draw_on_image(rgb, self.board, ratio)
        return rgb

    def mouse_key_press_draw(self, *args):
        rgb, ratio, event = args
        self._draw_on_image(rgb, self.board, ratio)
        return rgb

    def input_once_draw(self, *args):
        rgb, ratio = args
        self._draw_on_image(rgb, self.board, ratio)
        return rgb

    def redraw_on(self, *args):
        rgb, ratio = args
        self._draw_on_image(rgb, self.board, ratio)
        return rgb

    def get_results(self, *args, **kwargs):
        return [self.board]

    def valid(self) -> bool:
        return True

    def set_color(self, color):
        self.color = color



class Rectangle(Mark):
    # static variable
    __Name__ = 'rectangle'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.thick = 4
        self.color = (255, 0, 0)
        # for current
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None

    def __str__(self):
        return Rectangle.__Name__

    @staticmethod
    def name():
        return Rectangle.__Name__

    def _quantify_coordinate(self, x0, y0, x1, y1, ratio):
        x0 = self.quantify(min(x0, x1) * ratio)
        y0 = self.quantify(min(y0, y1) * ratio)
        x1 = self.quantify(max(x0, x1) * ratio)
        y1 = self.quantify(max(y0, y1) * ratio)
        return x0, y0, x1, y1

    def _draw_on(self, *args):
        rgb, x0, y0, x1, y1, ratio = args
        x0, y0, x1, y1 = self._quantify_coordinate(x0, y0, x1, y1, ratio)
        cv2.rectangle(rgb, (x0, y0), (x1, y1), color=self.color, thickness=int(self.thick/ratio))

    def mouse_press_event(self, *args):
        x, y, ratio = args
        x0, y0 = x / ratio, y / ratio
        self.x0 = x0
        self.y0 = y0

    def mouse_move_event(self, *args):
        x, y, ratio = args
        x1, y1 = x / ratio, y / ratio
        self.x1 = x1
        self.y1 = y1

    def mouse_release_event(self, *args):
        x, y, ratio = args
        x1, y1 = x / ratio, y / ratio
        self.x1 = x1
        self.y1 = y1
        x0 = min(self.x0, self.x1)
        y0 = min(self.y0, self.y1)
        x1 = max(self.x0, self.x1)
        y1 = max(self.y0, self.y1)
        self.history.append((x0, y0, x1, y1))

    def wheel_event(self, *args):
        pass

    def input_once_event(self, rect:List[int]):
        if len(rect) == 4:
            x0, y0, x1, y1 = rect
            self.x0, self.y0 = x0, y0
            self.x1, self.y1 = x1, y1
            self.history.append((x0, y0, x1, y1))
            return True
        return None

    def mouse_press_draw(self, *args):
        rgb, ratio, event = args
        return rgb

    def mouse_move_draw(self, *args):
        rgb, ratio, event = args
        copy = rgb.copy()
        self._draw_on(copy, self.x0, self.y0, self.x1, self.y1, ratio)
        return copy

    def mouse_release_draw(self, *args):
        rgb, ratio, event = args
        self._draw_on(rgb, self.x0, self.y0, self.x1, self.y1, ratio)
        return rgb

    def input_once_draw(self, *args):
        rgb, ratio = args
        self._draw_on(rgb, self.x0, self.y0, self.x1, self.y1, ratio)
        return rgb

    def redraw_on(self, *args):
        rgb, ratio = args
        for (x0, y0, x1, y1) in self.history:
            self._draw_on(rgb, x0, y0, x1, y1, ratio)

    def get_results(self, *args, **kwargs):
        data = []
        for x0, y0, x1, y1 in self.history:
            data.append(self._quantify_coordinate(x0, y0, x1, y1, 1.))
        return data

    def valid(self) -> bool:
        return bool(len(self.history) > 0)



class Polygon(Mark):
    # static variable
    __Name__ = 'polygon'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.thick = 4
        self.color = (255, 0, 0)
        self.board = np.zeros(self.size, dtype=np.uint8)
        # for current
        self.x, self.y = None, None
        self.sequential = []

    def __str__(self):
        return Polygon.__Name__

    @staticmethod
    def name():
        return Polygon.__Name__

    def _map_color(self, color):
        if tuple(color) == (255, 0, 0):
            return (2, 2, 2)
        if tuple(color) == (0, 0, 255):
            return (1, 1, 1)
        raise NotImplementedError

    def _quantify_coordinate(self, x, y, ratio, include):
        data = []
        all_points = self.sequential + ([(x, y),] if include == True else [])
        for xx, yy in all_points:
            xx = self.quantify(xx * ratio)
            yy = self.quantify(yy * ratio)
            data.append((xx, yy))
        return np.reshape(np.array(data, dtype=np.int32), (-1, 2))

    def _draw_on(self, *args):
        rgb, x, y, ratio, board, include = args
        points = self._quantify_coordinate(x, y, ratio, include)
        # visual convex face
        if len(points) >= 3:
            from skimage.draw import polygon2mask
            mask = polygon2mask(rgb.shape[:2], points[:,::-1]).astype(np.uint8)
            board[mask > 0] = self._map_color(self.color)[0]
        rgb_mask = np.ones_like(rgb)
        rgb_mask[board == 2, 1:] = 0
        rgb_mask[board == 1, :2] = 0
        rgb *= rgb_mask
        # visual outer lines
        if len(points) >= 2:
            cv2.polylines(rgb, [np.reshape(points, (-1,1,2))],
                isClosed=True, color=self.color, thickness=3)

    def _merge_into_mask(self):
        pass

    def mouse_press_event(self, *args):
        x, y, event, ratio = args
        if 'left-button' in event:
            self.x, self.y = x / ratio, y / ratio

    def mouse_move_event(self, *args):
        x, y, event, ratio = args
        self.x, self.y = x / ratio, y / ratio

    def mouse_release_event(self, *args):
        x, y, event, ratio = args
        if 'left-button' in event:
            xx, yy = x / ratio, y / ratio
            self.sequential.append((xx, yy))

    def mouse_key_press_event(self, *args):
        event, ratio = args
        if 'key-escape' in event:
            self.x, self.y = None, None
            self.sequential.clear()

    def input_once_event(self, *args, **kwargs):
        mask, mode = kwargs['mask'], kwargs['mode']
        if len(mask.shape) == 2:
            if mode == 'overwrite':
                self.board = np.copy(mask)
            if mode == 'select':
                self.board[mask > 0] = mask[mask > 0]
            return True
        return None

    def mouse_press_draw(self, *args):
        rgb, ratio, event = args
        self._draw_on(rgb, self.x, self.y, ratio, self.board.copy(), True)
        return rgb

    def mouse_move_draw(self, *args):
        rgb, ratio, event = args
        copy = rgb.copy()
        self._draw_on(copy, self.x, self.y, ratio, self.board.copy(), True)
        return copy

    def mouse_release_draw(self, *args):
        rgb, ratio, event = args
        self._draw_on(rgb, self.x, self.y, ratio, self.board, False)
        return rgb

    def mouse_key_press_draw(self, *args):
        rgb, ratio, event = args
        self._draw_on(rgb, self.x, self.y, ratio, self.board, False)
        return rgb

    def input_once_draw(self, *args):
        rgb, ratio = args
        self._draw_on(rgb, self.x, self.y, ratio, self.board, False)
        return rgb

    def redraw_on(self, *args):
        rgb, ratio = args
        self._draw_on(rgb, self.x, self.y, ratio, self.board, False)
        return rgb

    def get_results(self, *args, **kwargs):
        return [self.board]

    def valid(self) -> bool:
        return True

    def set_color(self, color):
        self.color = color



class MarkClassDict:
    def __init__(self, key: [], value: []):
        self.str2class_dict = dict(zip(key, value))
        self.class2str_dict = dict(zip(value, key))

    def __getitem__(self, item):
        return self.str2class_dict[item] if isinstance(item, str) \
            else self.class2str_dict[item]

    def string_dict(self, obj):
        string_dict = dict()
        for s in self.str2class_dict:
            string_dict[s] = obj.copy()
        return string_dict

    def class_dict(self, obj):
        class_dict = dict()
        for s in self.class2str_dict:
            class_dict[s] = obj.copy()
        return class_dict



class ColorMap:
    def __init__(self):
        pass

    @staticmethod
    def get_color_map(N=256, normalized=False):
        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)

        dtype = 'float32' if normalized else 'uint8'
        cmap = np.zeros((N, 3), dtype=dtype)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7 - j)
                g = g | (bitget(c, 1) << 7 - j)
                b = b | (bitget(c, 2) << 7 - j)
                c = c >> 3
            cmap[i] = np.array([r, g, b])

        cmap = cmap / 255 if normalized else cmap
        return cmap