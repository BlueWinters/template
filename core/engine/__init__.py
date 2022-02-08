
import os
import platform
import logging


EngineClassDict = dict()
GlobalEngine = os.environ.get('ENGINE', None)

def import_engine_torch():
    from .libengine_th import LibEngineTorch
    EngineClassDict['torch'] = LibEngineTorch
    logging.info('import engine torch')

def import_engine_tonsorrt():
    from .libengine_trt import LibEngineTensorRT
    EngineClassDict['tensorrt'] = LibEngineTensorRT
    logging.info('import engine tensorrt')

def import_engine_ncnn():
    from .libengine_ncnn import LibEngineNcnn
    EngineClassDict['ncnn'] = LibEngineNcnn
    logging.info('import engine ncnn')

def import_engine_online():
    from .libengine_online import LibEngineOnline
    EngineClassDict['online'] = LibEngineOnline
    logging.info('import engine online')


if GlobalEngine is None:
    if platform.system().lower() == 'linux':
        import_engine_torch()
        import_engine_tonsorrt()
        import_engine_online()
        import_engine_ncnn()
    if platform.system().lower() == 'windows':
        import_engine_torch()
        # import_engine_online()
        # import_engine_ncnn()
else:
    if GlobalEngine.lower() == 'torch':
        import_engine_torch()
    if GlobalEngine.lower() == 'tensorrt':
        import_engine_tonsorrt()
    if GlobalEngine.lower() == 'ncnn':
        import_engine_ncnn()


def create_engine(engine_config:dict):
    if GlobalEngine is None:
        EngineClass = EngineClassDict[engine_config['type']]
        return EngineClass.create(config=engine_config)
    else:
        engine_config['type'] = str(GlobalEngine).lower()
        EngineClass = EngineClassDict[engine_config['type']]
        return EngineClass.create(config=engine_config)