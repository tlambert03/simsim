from pycuda.compiler import SourceModule
import pycuda.driver as cuda
from pycuda import gpuarray
import numpy as np
import os

import pycuda.autoinit  # noqa

cubic_dir = os.path.join(os.path.dirname(__file__), "cubic")
with open(__file__.replace(".py", ".cu"), "r") as f:
    mod_affine = SourceModule(f.read(), no_extern_c=True, include_dirs=[cubic_dir])

_affine2D = mod_affine.get_function("affine2D")
_affine2D_RA = mod_affine.get_function("affine2D_RA")
_affine3D = mod_affine.get_function("affine3D")
_affine3D_RA = mod_affine.get_function("affine3D_RA")
texref2D = mod_affine.get_texref("texref2d")
texref3D = mod_affine.get_texref("texref3d")
texref2D.set_address_mode(0, cuda.address_mode.BORDER)
texref2D.set_address_mode(1, cuda.address_mode.BORDER)
texref3D.set_address_mode(0, cuda.address_mode.BORDER)
texref3D.set_address_mode(1, cuda.address_mode.BORDER)
texref3D.set_address_mode(2, cuda.address_mode.BORDER)


s2c2dx = None
s2c2dy = None
s2c3dx = None
s2c3dy = None
s2c3dz = None


def import_prefilter_2D():
    global s2c2dx, s2c2dy
    with open(os.path.join(cubic_dir, "cubicPrefilter2D.cu"), "r") as f:
        modcubic2 = SourceModule(f.read(), no_extern_c=True, include_dirs=[cubic_dir])
    s2c2dx = modcubic2.get_function("SamplesToCoefficients2DX")
    s2c2dy = modcubic2.get_function("SamplesToCoefficients2DY")


def import_prefilter_3D():
    global s2c3dx, s2c3dy, s2c3dz
    with open(os.path.join(cubic_dir, "cubicPrefilter3D.cu"), "r") as f:
        modcubic3 = SourceModule(f.read(), no_extern_c=True, include_dirs=[cubic_dir])
    s2c3dx = modcubic3.get_function("SamplesToCoefficients3DX")
    s2c3dy = modcubic3.get_function("SamplesToCoefficients3DY")
    s2c3dz = modcubic3.get_function("SamplesToCoefficients3DZ")


def _bind_tex(array):
    assert array.ndim in (2, 3), "Texture binding only valid for 2 or 3D arrays"
    if isinstance(array, np.ndarray):
        ary = cuda.np_to_array(array, "F" if np.isfortran(array) else "C")
    elif isinstance(array, gpuarray.GPUArray):
        ary = cuda.gpuarray_to_array(array, "F" if array.flags.f_contiguous else "C")
    else:
        raise ValueError("Can only bind numpy arrays or GPUarray")
    if array.ndim == 2:
        texref2D.set_array(ary)
    elif array.ndim == 3:
        texref3D.set_array(ary)


def _set_tex_filter_mode(mode, ndim):
    assert mode in [
        "linear",
        "point",  # aka nearest neighbor
        "nearest",
        "cubic",
        "cubic-prefilter",
    ], f"unrecognized interpolation mode: {mode}"
    if mode == "linear":
        # default is point
        if ndim == 3:
            texref3D.set_filter_mode(cuda.filter_mode.LINEAR)
        elif ndim == 2:
            texref2D.set_filter_mode(cuda.filter_mode.LINEAR)


def _with_bound_texture(func):
    def wrapper(*args, **kwargs):
        args = list(args)
        array = args[0]
        # so far these all require float32
        if not array.dtype == np.float32:
            array = array.astype(np.float32)
        kmode = kwargs.get("mode", "")
        if ("pre" in kmode and "cub" in kmode) or any(
            [("pre" in x and "cub" in x) for x in args if isinstance(x, str)]
        ):
            array = spline_filter(array)
        # bind array to textureRef
        _bind_tex(array)
        args[0] = array
        _set_tex_filter_mode(kwargs.get("mode", "nearest"), ndim=array.ndim)
        return func(*args, **kwargs)

    return wrapper


def make_translation_matrix(mag):
    if len(mag) == 3:
        tmat = np.eye(4)
        tmat[0, 3] = mag[2]
        tmat[1, 3] = mag[1]
        tmat[2, 3] = mag[0]
    elif len(mag) == 2:
        tmat = np.eye(3)
        tmat[0, 2] = mag[1]
        tmat[1, 2] = mag[0]
    return tmat


def make_scaling_matrix(scalar):
    if len(scalar) == 3:
        tmat = np.eye(4)
        tmat[0, 0] = 1 / scalar[2]
        tmat[1, 1] = 1 / scalar[1]
        tmat[2, 2] = 1 / scalar[0]
    elif len(scalar) == 2:
        tmat = np.eye(3)
        tmat[0, 0] = 1 / scalar[1]
        tmat[1, 1] = 1 / scalar[0]
    return tmat


def make_rotation_matrix(array, angle, axis=0):
    theta = angle * np.pi / 180
    _sin = np.sin(theta)
    _cos = np.cos(theta)
    if array.ndim == 3:
        nz, ny, nx = array.shape
        # first translate the middle of the image to the origin
        T1 = np.array(
            [[1, 0, 0, nx / 2], [0, 1, 0, ny / 2], [0, 0, 1, nz / 2], [0, 0, 0, 1]]
        )
        # then rotate theta degrees about the Y axis
        if axis in (0, "z", "Z"):
            R = np.array(
                [[_cos, _sin, 0, 0], [-_sin, _cos, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            )
        elif axis in (1, "y", "Y"):
            R = np.array(
                [[_cos, 0, -_sin, 0], [0, 1, 0, 0], [_sin, 0, _cos, 0], [0, 0, 0, 1]]
            )
        elif axis in (2, "x", "X"):
            R = np.array(
                [[1, 0, 0, 0], [0, _cos, _sin, 0], [0, -_sin, _cos, 0], [0, 0, 0, 1]]
            )
        else:
            raise ValueError("Unrecognized axis of rotation: {}".format(axis))
        # then translate back to the original origin
        T2 = np.array(
            [[1, 0, 0, -nx / 2], [0, 1, 0, -ny / 2], [0, 0, 1, -nz / 2], [0, 0, 0, 1]]
        )
        return np.dot(np.dot(np.dot(np.eye(4), T1), R), T2)
    if array.ndim == 2:
        ny, nx = array.shape
        # first translate the middle of the image to the origin
        T1 = np.array([[1, 0, nx / 2], [0, 1, ny / 2], [0, 0, 1]])
        # then rotate theta degrees
        R = np.array([[_cos, -_sin, 0], [_sin, _cos, 0], [0, 0, 1]])
        # then translate back to the original origin
        T2 = np.array([[1, 0, -nx / 2], [0, 1, -ny / 2], [0, 0, 1]])
        return np.dot(np.dot(np.dot(np.eye(3), T1), R), T2)
    raise ValueError("Can only do 2D and 3D rotations")


def _make_grid(shape, blocks):
    if len(shape) == 3:
        out_z, out_y, out_x = shape
        bx, by, bz = blocks
        return ((out_x + bx - 1) // bx, (out_y + by - 1) // by, (out_z + bz - 1) // bz)
    elif len(shape) == 2:
        out_y, out_x = shape
        bx, by = blocks[:2]
        return ((out_x + bx - 1) // bx, (out_y + by - 1) // by, 1)


def _do_affine(shape, tmat, mode, blocks):
    if len(shape) == 3:
        if not tmat.shape == (4, 4):
            raise ValueError(f"3D transformation matrix must be 4x4, saw {tmat.shape}")
        _func = _affine3D
        _tref = texref3D
    elif len(shape) == 2:
        if not tmat.shape == (3, 3):
            raise ValueError(f"3D transformation matrix must be 3x3, saw {tmat.shape}")
        _func = _affine2D
        _tref = texref2D

    output = gpuarray.empty(shape, dtype=np.float32)
    grid = _make_grid(shape, blocks)
    _func(
        output,
        *np.flip(np.int32(output.shape)),
        cuda.In(tmat.astype(np.float32).ravel()),
        np.int32("cubic" in mode.lower()),
        texrefs=[_tref],
        block=blocks,
        grid=grid,
    )
    return output


@_with_bound_texture
def zoom(input, zoom, mode="nearest", blocks=(16, 16, 4)):
    """scale array with nearest neighbors or linear interpolation

    If a float, `zoom` is the same for each axis. If a sequence,
    `zoom` should contain one value for each axis.
    """
    if isinstance(zoom, (int, float)):
        zoom = tuple([zoom] * input.ndim)
    assert (
        len(zoom) == input.ndim
    ), "scalar must either be a scalar or a list with the same length as array.ndim"

    # make scaling array
    tmat = make_scaling_matrix(zoom)
    outshape = tuple(int(x) for x in np.round(np.array(input.shape) * zoom))
    return _do_affine(outshape, tmat, mode, blocks)


@_with_bound_texture
def rotate(input, angle, axis=0, mode="nearest", blocks=(16, 16, 4)):
    """Rotate an array.

    axis can be either 0,1,2 or z,y,x
    """
    tmat = make_rotation_matrix(input, angle, axis)
    return _do_affine(input.shape, tmat, mode, blocks)


@_with_bound_texture
def shift(input, shift, mode="nearest", blocks=(16, 16, 4)):
    """translate array

    mag is number of pixels to translate in (z,y,x)
    must be tuple with length array.ndim
    """
    if isinstance(shift, (int, float)):
        shift = tuple([shift] * input.ndim)
    assert (
        len(shift) == input.ndim
    ), "shift must either be a scalar or a list with the same length as array.ndim"

    tmat = make_translation_matrix(shift)
    return _do_affine(input.shape, tmat, mode, blocks)


@_with_bound_texture
def affine_transform(input, matrix, mode="nearest", blocks=(16, 16, 4)):
    """Apply an affine transformation.

    Args:
        input (pycuda.gpuarray): The input array.
        matrix (pycuda.gpuarray): The inverse coordinate transformation matrix,
            mapping output coordinates to input coordinates. If ``ndim`` is the
            number of dimensions of ``input``, the given matrix must be of shape
            ``(ndim + 1, ndim + 1)``: (assume that the transformation is
            specified using homogeneous coordinates).
        mode (str): type of interpolation ('nearest', 'linear', 'cubic', 'cubic-prefilter')
    Returns:
        pycuda.gpuarray
    .. seealso:: :func:`scipy.ndimage.affine_transform`
    """
    return _do_affine(input.shape, matrix, mode, blocks)


def pow2divider(num):
    if num == 0:
        return 0
    divider = 1
    while (num & divider) == 0:
        divider <<= 1
    return divider


def _cubic_bspline_prefilter_3D(ary_gpu):
    if s2c3dx is None:
        import_prefilter_3D()
    depth, height, width = np.int32(ary_gpu.shape)
    pitch = np.int32(width * 4)  # width of a row in the image in bytes
    dimX = np.int32(min(min(pow2divider(width), pow2divider(height)), 64))
    dimY = np.int32(min(min(pow2divider(depth), pow2divider(height)), 512 / dimX))
    blocks = (int(dimX), int(dimY), 1)
    gridX = (int(height // dimX), int(depth // dimY), 1)
    gridY = (int(width // dimX), int(depth // dimY), 1)
    gridZ = (int(width // dimX), int(height // dimY), 1)
    s2c3dx(ary_gpu, pitch, width, height, depth, block=blocks, grid=gridX)
    s2c3dy(ary_gpu, pitch, width, height, depth, block=blocks, grid=gridY)
    s2c3dz(ary_gpu, pitch, width, height, depth, block=blocks, grid=gridZ)
    return ary_gpu


def _cubic_bspline_prefilter_2D(ary_gpu):
    if s2c2dx is None:
        import_prefilter_2D()
    height, width = np.int32(ary_gpu.shape)
    pitch = np.int32(width * 4)  # width of a row in the image in bytes
    blockx = (int(min(pow2divider(height), 64)), 1, 1)
    blocky = (int(min(pow2divider(width), 64)), 1, 1)
    gridX = (int(height // blockx[0]), 1, 1)
    gridY = (int(width // blocky[0]), 1, 1)
    s2c2dx(ary_gpu, pitch, width, height, block=blockx, grid=gridX)
    s2c2dy(ary_gpu, pitch, width, height, block=blocky, grid=gridY)
    return ary_gpu


def spline_filter(array):
    if not isinstance(array, gpuarray.GPUArray):
        ary_gpu = gpuarray.to_gpu(np.ascontiguousarray(array).astype(np.float32))
    if array.ndim == 2:
        return _cubic_bspline_prefilter_2D(ary_gpu)
    elif array.ndim == 3:
        return _cubic_bspline_prefilter_3D(ary_gpu)
