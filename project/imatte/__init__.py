
from .libgca import LibGCA
from .libsim import LibSIM
from .libtimi import LibTIMI
from .libdim import LibDIM
from .libfgi import LibFGI


LibClassDict = dict(
    gca=LibGCA,
    sim=LibSIM,
    timi=LibTIMI,
    dim=LibDIM,
    fgi=LibFGI,
)


class LibPackage:
    def __init__(self, models: dict):
        self.models = models

    def __iter__(self):
        return iter(self.models)

    def initialize(self):
        for name in self.models: self.models[name].initialize()

    def pipeline(self, *args, **kwargs):
        name = kwargs.pop('model') if 'model' in kwargs else ''
        models = [name] if name in self.models else list(self.models.keys())
        return {m: self.models[m].pipeline(
            *args, **kwargs) for m in models}



def create_instance(config:dict):
    assert 'matte' in config
    name = config['matte']
    all_names = [name] if name in LibClassDict else list(LibClassDict.keys())
    models = {name:LibClassDict[name](config[name]) for name in all_names}
    return LibPackage(models)
