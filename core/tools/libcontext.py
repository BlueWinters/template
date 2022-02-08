import os
import time


class LibContexTemplate:
    def __init__(self, type):
        self.type = type
        self.value = 0
        self.operator = dict(
            maximum=lambda beg, end: self.maximum(beg, end),
            common=lambda beg, end: (end - beg),
        )[self.type]

    def record(self):
        ...

    def maximum(self, beg, end):
        self.value = max(self.value, end - beg)
        return self.value

    def measure(self, beg, end):
        return self.operator(beg, end)

    def direct(self):
        return bool(self.type == 'common')

    def format(self, value):
        ...



class LibContexTimer(LibContexTemplate):
    def __init__(self, type='common'):
        super(LibContexTimer, self).__init__(type)

    def record(self):
        return time.time()

    def format(self, eclipse):
        header = 'time running({}):'.format(self.type).ljust(30)
        return '{} {:.4f} ms'.format(header, eclipse * 1000)



class LibContexRAMMemory(LibContexTemplate):
    def __init__(self, type='common'):
        super(LibContexRAMMemory, self).__init__(type)

    def record(self):
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss

    def format(self, num_bytes):
        header = 'RAM memory changes({}):'.format(self.type).ljust(30)
        return '{} {:.4f} GB'.format(header, num_bytes / 1024 / 1024 / 1024)



class LibContexGPUMemory(LibContexTemplate):
    def __init__(self, type='common'):
        super(LibContexGPUMemory, self).__init__(type)

    def record(self):
        # import pycuda.autoinit
        import pycuda.driver as driver
        free, total = driver.mem_get_info()
        return total - free

    def format(self, num_bytes):
        header = 'GPU memory changes({}):'.format(self.type).ljust(30)
        return '{} {:.4f} GB'.format(header, num_bytes / 1024 / 1024 / 1024)



class LibContext:
    def __init__(self, *context):
        self.contexts = context

    def __del__(self):
        print(LibContext.code_information(self.function))
        for ctx in self.contexts:
            if ctx.direct() is False:
                line = ctx.format(ctx.value)
                print('\t{}'.format(line))

    def __call__(self, function):
        self.function = function
        def call_wrapper(*args, **kwargs):
            return LibContext.measure_context(self.function, self.contexts, *args, **kwargs)
        return call_wrapper

    @staticmethod
    def code_information(function):
        code = function.__code__
        line = '{}::line_{}::{}'.format(
            code.co_filename, code.co_firstlineno, code.co_name)
        return line

    @staticmethod
    def measure_context(function, context, *args, **kwargs):
        pre = list(map(lambda ctx: ctx.record(), context))
        output = function(*args, **kwargs)
        aft = list(map(lambda ctx: ctx.record(), context))
        print(LibContext.code_information(function))
        for ctx, p, a in zip(context, pre, aft):
            line = ctx.format(ctx.measure(p, a))
            if ctx.direct() is True:
                print('\t{}'.format(line))
        return output

    @staticmethod
    def get_context(timer:str='common', ram:str='common', gpu:str='common'):
        return LibContext(
            LibContexTimer(timer),
            LibContexRAMMemory(ram),
            LibContexGPUMemory(gpu)
        )
