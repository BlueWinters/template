

class LibCore:
    """
    use for manage objects and configuration
    """
    _instance = None

    @staticmethod
    def _load_yaml(yaml_path: str):
        import os, yaml
        if os.path.exists(yaml_path) is False:
            print('load config fail: {}'.format(yaml_path))
        config = yaml.load(open(yaml_path, 'r'), Loader=yaml.Loader)
        return config

    @staticmethod
    def _create_object(config:dict):
        from .imatte import create_instance
        return create_instance(config)

    @staticmethod
    def _global_config():
        import logging, platform
        if platform.system().lower() == 'windows':
            logging.basicConfig(level=logging.INFO)
            return 'project/imatte/config.yaml'
        raise NotImplementedError

    """
    """
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self):
        yaml_path = LibCore._global_config()
        self.config = LibCore._load_yaml(yaml_path)
        self.object = LibCore._create_object(self.config)

    def initialize(self):
        pass
        self.object.initialize()

    def pipeline(self, *args, **kwargs):
        return self.object.pipeline(*args, **kwargs)
