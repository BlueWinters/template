
import numpy as np
import json
import requests
import logging
from typing import Dict, Tuple, List, Optional
from .libengine import LibEngine
from ..tools.libarray import LibArray


class LibEngineOnline(LibEngine):
    """
    """
    @staticmethod
    def create(*args, **kwargs):
        return LibEngineOnline(config=kwargs['config'])

    """
    """
    def __init__(self, config:Dict):
        super(LibEngineOnline, self).__init__(config)
        self.type = 'online'
        self.url = config['url']

    """
    """
    def initialize(self, *args, **kwargs):
        pass

    """
    """
    def inference(self, inputs:Tuple[np.ndarray,...], *args, **kwargs):
        assert len(inputs) > 0
        try:
            inputs_encode = self._convert_inputs(inputs)
            response = requests.post(url=self.url, data={'inputs':inputs_encode})
            outputs_decode = self._convert_outputs(response)
            return outputs_decode
        except requests.exceptions.RequestException as e:
            logging.error('http post error {}'.format(e))

    def _convert_inputs(self, inputs:Tuple[np.ndarray,...]):
        encode = LibArray.encode(inputs)
        return json.dumps(encode)

    def _convert_outputs(self, response:requests.Response):
        outputs = json.loads(str(response.content, 'utf-8'))
        if outputs['code'] == 1:
            logging.error('process exception with code 400')
            return None
        return LibArray.decode(outputs['data'])