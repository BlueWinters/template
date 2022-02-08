
import os
import logging
import ncnn
import numpy as np
from typing import Dict, Tuple, List, Optional
from .libengine import LibEngine


class LibEngineNcnn(LibEngine):
    def __init__(self, config: Dict):
        super(LibEngineNcnn, self).__init__(config)
        self.type = 'ncnn'
        self.use_gpu = self.option('use_gpu', False)
        self.num_threads = self.option('num_threads', 1)

    def __del__(self):
        self.module = None

    """
    """
    @staticmethod
    def create(*args, **kwargs):
        return LibEngineNcnn(config=kwargs['config'])

    """
    """
    def initialize(self, *args, **kwargs):
        self.module = LibEngineNcnn._load(
            *self._parse(self.config['parameters']))
        # config about GPU
        self.module.opt.use_vulkan_compute = self.use_gpu

    def _parse(self, parameters:[str,List,Tuple]):
        if isinstance(parameters, str):
            name, _ = os.path.splitext(parameters)
            path_param = '{}.param'.format(name)
            path_model = '{}.bin'.format(name)
            return path_param, path_model
        if isinstance(parameters, (list, tuple)):
            return parameters[0], parameters[1]

    @staticmethod
    def _load(path_param:str, path_model:str):
        assert os.path.exists(path_param)
        assert os.path.exists(path_model)
        net = ncnn.Net()
        param_code = net.load_param(path_param)
        model_code = net.load_model(path_model)
        assert (param_code == 0) and (model_code == 0)
        logging.info('load model: {} & {}'.format(path_param, path_model))
        return net

    """
    """
    def inference(self, *args, **kwargs):
        assert hasattr(self, 'module')
        assert len(kwargs) > 0 or 'inputs' in kwargs
        ex = self._create_extractor()
        self._assign_inputs(ex, **kwargs)
        return self._extract_outputs(ex)

    def _create_extractor(self, **kwargs):
        ex = self.module.create_extractor()
        ex.set_num_threads(self.num_threads)
        return ex

    def _assign_inputs(self, ex, **kwargs):
        """
        input format:
            1. kwargs: only one input in kwargs with a dict {string: array}
            2. kwargs: inputs with its name
        input example:
            1. self.engine(inputs={'name1':input1, 'name2':input2})
            2. self.engine(name1=input1, name2=input2)
        data order:
            format: (N,C,H,W) or (C,H,W)
        """
        if len(kwargs) > 0:
            inputs_dict = kwargs['inputs'] if \
                'inputs' in kwargs and isinstance(kwargs['inputs'], dict) \
                else kwargs
            for name in self.module.input_names():
                assert isinstance(name, str)
                assert name in inputs_dict
                ex.input(name, self._convert_array2mat(inputs_dict[name]))

    def _convert_array2mat(self, array:np.ndarray):
        if len(array.shape) == 4:
            n, c, h, w = array.shape
            assert n == 1
            array = np.squeeze(array, axis=0)
        return ncnn.Mat(np.ascontiguousarray(array, dtype=np.float32))

    def _extract_outputs(self, ex):
        outputs = dict()
        for name in self.module.output_names():
            ret, mat = ex.extract(name)
            assert mat.elemsize == 4  # ref: https://github.com/Tencent/ncnn/blob/master/python/src/main.cpp#L307
            outputs[name] = np.array(mat)
        return outputs