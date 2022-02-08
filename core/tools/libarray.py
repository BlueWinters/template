
import io
import numpy as np
import base64
from typing import Tuple, Dict


class LibArray:
    """
    """
    DataTypeDictPair = (
        ('uint8', np.uint8),
        ('float32', np.float32),
        ('double', np.double),
    )
    DataTypeDict_S2C = {p[0]:p[1] for p in DataTypeDictPair}
    DataTypeDict_C2S = {p[1]:p[0] for p in DataTypeDictPair}

    """
    """
    @staticmethod
    def encode(inputs:Tuple[np.ndarray,...]) -> Dict:
        data, shape, size, type = io.BytesIO(), list(), list(), list()
        for n, array in enumerate(inputs):
            num = data.write(array.tobytes())
            shape.append(array.shape)
            size.append(num)
            type.append(LibArray._type(array.dtype.type))
        encode_data = base64.b64encode(data.getvalue()).decode('utf-8')
        return dict(data=encode_data, shape=shape, size=size, type=type)

    @staticmethod
    def _type(type):
        return LibArray.DataTypeDict_S2C[type] if isinstance(type, str) \
            else LibArray.DataTypeDict_C2S[type]

    """
    """
    @staticmethod
    def decode(inputs:Dict) -> Tuple[np.ndarray,...]:
        all_data = base64.b64decode(inputs['data'])
        all_shape = inputs['shape']
        all_size = inputs['size']
        all_type = inputs['type']

        assert len(all_shape) == len(all_size)
        array_list, count = list(), 0
        for shape, size, type in zip(all_shape, all_size, all_type):
            buffer = all_data[count:count+size]
            array = np.frombuffer(buffer, dtype=LibArray._type(type))
            array_list.append(np.reshape(array, shape))
            count += size
        return tuple(array_list)

    """
    """
    @staticmethod
    def benchmark():
        data1 = np.random.uniform(0, 1, (3, 5)).astype(np.double)
        data2 = np.random.uniform(0, 1, (3, 5)).astype(np.float32)
        print(data1, data1[0].dtype)
        print(data2, data2[0].dtype)
        encode_data = LibArray.encode(inputs=(data1, data2))
        decode_data = LibArray.decode(encode_data)
        print(decode_data[0], decode_data[0].dtype)
        print(decode_data[1], decode_data[1].dtype)