from simsim.cuda import initpycuda
from pycuda.compiler import SourceModule
import numpy as np
import pycuda.driver as cuda
from pycuda import gpuarray


with open(__file__.replace(".py", ".cu"), "r") as f:
    mod = SourceModule(f.read())
_bresenham3D = mod.get_function("bresenham3D")


def bresenham_3D(vertices, max_out=512):
    """
    vertices should be a numpy array with shape (n, 6)
    [[x1, y1, z1, x2, y2, z2],
     [x1, y1, z1, x2, y2, z2],
     ...,
     [x1, y1, z1, x2, y2, z2]]
    """
    if not (vertices.ndim == 2 and vertices.shape[1] == 6):
        raise ValueError("vertices should be a numpy array with shape (n, 6)")

    numpairs = len(vertices)
    vert_gpu = gpuarray.to_gpu(np.ascontiguousarray(vertices.astype(np.int32)))
    out = np.ones((max_out, numpairs, 3), dtype=np.int32, order="C") * -1
    _bresenham3D(
        np.int32(numpairs),
        vert_gpu,
        cuda.InOut(out),
        block=(256, 1, 1),
        grid=(numpairs // 256, 1, 1),
    )
    return out
