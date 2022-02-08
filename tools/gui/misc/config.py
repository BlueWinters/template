
from typing import Dict


class Configuration:
    def __init__(self, config:Dict):
        self.data = self._check_config(config)

    def _check_config(self, config:Dict):
        data = dict()
        for key, value in config.items():
            assert isinstance(value, dict)
            assert 'type' in value
            assert 'default' in value
            assert isinstance(value['default'], value['type'])
            data[key] = dict(
                # function=lambda s: value['type'](s),
                type=value['type'],
                default=value['default'],
                current=value['default'],
                checker=value['checker'] if 'checker' in value else None,
            )
        return data

    def rollback(self, key:str):
        self.data[key]['current'] = self.data[key]['default']

    def pair(self) -> Dict:
        return {key: value['current'] for key, value in self.data.items()}

    def package(self) -> Dict:
        return {key: (value['type'], value['current']) for key, value in self.data.items()}

    def __getitem__(self, key:str):
        return self.data[key]['current']

    def __setitem__(self, key:str, value):
        try:
            value = self.data[key]['type'](value)
            if self.data[key]['checker'] is not None:
                assert self.data[key]['checker'](value)
            self.data[key]['current'] = value
        except ValueError as e:
            print('config value error in {} with {}: reset to default {}'.format(
                key, value, self.data[key]['default']))
            self.data[key]['current'] = self.data[key]['default']
        # except AssertionError as e:
        #     print('config assert error in {} with {}: reset to default {}'.format(
        #         key, value, self.data[key]['default']))
        #     self.data[key]['current'] = self.data[key]['default']

    def __iter__(self):
        return {key:value['current'] for key, value in self.data.items()}.items().__iter__()

    def __len__(self):
        return len(self.data)