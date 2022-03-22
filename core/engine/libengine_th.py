
import os
import torch
import numpy as np
from typing import Dict, Tuple, List, Optional
from .libengine import LibEngine


class LibEngineTorch(LibEngine):
    """
    """
    @staticmethod
    def create(*args, **kwargs):
        return LibEngineTorch(config=kwargs['config'])

    def __init__(self, config:Dict):
        super(LibEngineTorch, self).__init__(config)
        self.type = 'torch'
        self.attach = lambda array: \
            torch.from_numpy(array).to(self.device).float() \
                if isinstance(array, np.ndarray) else array
        self.detach = lambda tensor: tensor.cpu().detach().numpy()

    def __del__(self):
        self.module, self.device = None, None

    """
    """
    def initialize(self, *args, **kwargs):
        self.device = LibEngineTorch._device(self.config['device'])
        self.module = LibEngineTorch._load(self.config['parameters'], self.device)

    @staticmethod
    def _device(device:str):
        assert 'cuda' in device or 'cuda' in device
        if torch.cuda.is_available() is False:
            # TODO: assert cuda is not available
            device = 'cpu'
        return torch.device(device)

    @staticmethod
    def _load(path:str, device:torch.device):
        assert os.path.exists(path)
        print('load model: ', path)
        net = torch.jit.load(path, map_location=device)
        return net.eval()

    """
    """
    def inference(self, *args, **kwargs):
        with torch.no_grad():
            assert hasattr(self, 'module')
            assert len(args) > 0 or 'inputs' in kwargs
            inputs = self._convert_inputs(*args, **kwargs)
            output = self.module(*inputs)
            return self._convert_outputs(output)

    def _convert_inputs(self, *args, **kwargs):
        """
        input format:
            1. args: a sequential inputs (input_1,...,input_n) with a list [array/tensor,...]
            2. args: only one input in kwargs with a dict {string: array/tensor}
        input example:
            1. self.engine(input1, input2, input3)
            2. self.engine(inputs={'name1':input1, 'name2':input2})
        """
        if len(args) > 0:
            tensor_list = list()
            for array in args:
                assert isinstance(array, (np.ndarray, torch.Tensor))
                tensor = self.attach(array)
                tensor_list.append(tensor)
            return tensor_list
        if isinstance(kwargs['inputs'], dict):
            inputs_dict = kwargs['inputs']
            tensor_dict = dict()
            for name, array in inputs_dict.items():
                assert isinstance(name, str)
                assert isinstance(array, (np.ndarray, torch.Tensor))
                tensor_dict[name] = self.attach(array)
            return [tensor_dict]
        raise ValueError

    def _convert_outputs(self, outputs):
        """
        output format:
            1. a single tensor
            2. list/tuple: a sequential outputs like (output_1,...,output_n)
            3. only one output with a dict {string: array/tensor}
        output example:
            1. return output
            2. return output1, output2, output2
            3. return {'name1':input1, 'name2':input2}
        """
        if isinstance(outputs, torch.Tensor):
            return self.detach(outputs)
        if isinstance(outputs, (list, tuple)):
            array_list = list()
            for out in outputs:
                if isinstance(out, torch.Tensor):
                    array_list.append(self.detach(out))
            return array_list
        if isinstance(outputs, dict):
            array_dict = dict()
            for name, tensor in outputs.items():
                assert isinstance(name, str)
                array_dict[name] = self.detach(tensor)
            return array_dict
        return ValueError
