from __future__ import absolute_import

import atexit

import pycuda.driver as cuda

# Initialize CUDA
cuda.init()


def get_best_gpu():
    devs = []
    for n, devn in enumerate(range(cuda.Device.count())):
        dev = cuda.Device(devn)
        devs.append((dev, dev.total_memory()))
    devs.sort(key=lambda x: -x[1])
    return devs[0][0]


global context
device = get_best_gpu()
context = device.make_context()


def _finish_up():
    global context
    context.pop()
    context = None
    from pycuda.tools import clear_context_caches

    clear_context_caches()


atexit.register(_finish_up)
