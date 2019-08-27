import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
from pycuda import gpuarray
import numpy as np
import logging

logger = logging.getLogger(__name__)

mod_affine = SourceModule(
    """
    //#include <stdio.h>

    texture<float, cudaTextureType3D, cudaReadModeElementType> texRef;

    // Simple transformation kernel
    __global__ void transformKernel(float *output, int nx, int ny, int nz,
                                    float *mat) {

        // Calculate texture coordinates
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x >= nx || y >= ny || z >= nz) {
            return;
        }

        float u = x;
        float v = y;
        float w = z;

        float tu = mat[0] * u + mat[1] * v + mat[2] * w + mat[3] + 0.5f;
        float tv = mat[4] * u + mat[5] * v + mat[6] * w + mat[7] + 0.5f;
        float tw = mat[8] * u + mat[9] * v + mat[10] * w + mat[11] + 0.5f;

        // Read from texture and write to global memory
        int idx = z * (nx * ny) + y * nx + x;
        output[idx] = tex3D(texRef, tu, tv, tw);

        // if (x - 1 < nx / 2 && x + 1 > nx / 2 && y - 1 < ny / 2 && y + 1 > ny / 2){
        //     printf("x: %d y: %d z: %d; tu: %f tv: %f tw: %f\\n", x, y, z, tu, tv, tw);
        // }

    }

    // Simple transformation kernel
    __global__ void transformKernelRA(float *output, int nx, int ny, int nz,
                                    float dx, float dy, float dz, float *mat) {

        // Calculate texture coordinates
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x >= nx || y >= ny || z >= nz) {
            return;
        }

        float u = x;
        float v = y;
        float w = z;

        // intrinsic coords to world
        u = 0.5 + (u - 0.5) * dx;
        v = 0.5 + (v - 0.5) * dy;
        w = 0.5 + (w - 0.5) * dz;

        // transform coordinates in world coordinate frame
        float tu = mat[0] * u + mat[1] * v + mat[2] * w + mat[3];
        float tv = mat[4] * u + mat[5] * v + mat[6] * w + mat[7];
        float tw = mat[8] * u + mat[9] * v + mat[10] * w + mat[11];

        // world coords to intrinsic
        tu = 0.5 + (tu - 0.5) / dx;
        tv = 0.5 + (tv - 0.5) / dy;
        tw = 0.5 + (tw - 0.5) / dz;

        // Read from texture and write to global memory
        int idx = z * (nx * ny) + y * nx + x;
        output[idx] = tex3D(texRef, tu, tv, tw);
    }
    """
)

_affine = mod_affine.get_function("transformKernel")
affineRA = mod_affine.get_function("transformKernelRA")
texref = mod_affine.get_texref("texRef")


def with_cupy_conversion(func):
    def wrapper(*args, **kwargs):
        args = list(args)
        array = args[0]
        if (
            f"{type(array).__module__}.{type(array).__name__}"
            == "cupy.core.core.ndarray"
        ):
            print("converting from cupy array to pycuda array")
            array = array.get()
            pycuda.autoinit.context.push()
            array = gpuarray.to_gpu(array)
            args[0] = array
            v = func(*args, **kwargs)
            pycuda.autoinit.context.pop()
            return v
        return func(*args, **kwargs)
    return wrapper


def _bind_tex(array):
    pop = False
    if f"{type(array).__module__}.{type(array).__name__}" == "cupy.core.core.ndarray":
        print("converting from cupy array to pycuda array")
        array = array.get()
        pycuda.autoinit.context.push()
        pop = True
    if isinstance(array, np.ndarray):
        ary = cuda.np_to_array(array, "F" if np.isfortran(array) else "C")
    elif isinstance(array, gpuarray.GPUArray):
        ary = cuda.gpuarray_to_array(array, "F" if array.flags.f_contiguous else "C")
    else:
        raise ValueError("Can only bind numpy arrays or GPUarray")
    texref.set_array(ary)
    if pop:
        pycuda.autoinit.context.pop()


def _set_tex_filter_mode(mode):
    assert mode in ["linear", "point", "nearest"], "unrecognized interpolation mode"
    if mode == "linear":
        # default is point
        texref.set_filter_mode(cuda.filter_mode.LINEAR)


def _make_translation_matrix(mag):
    tmat = np.eye(4)
    tmat[0, 3] = mag[2]
    tmat[1, 3] = mag[1]
    tmat[2, 3] = mag[0]
    return tmat


def _make_scaling_matrix(scalar):
    tmat = np.eye(4)
    tmat[0, 0] = 1 / scalar[2]
    tmat[1, 1] = 1 / scalar[1]
    tmat[2, 2] = 1 / scalar[0]
    return tmat


def _make_rotation_matrix(array, angle, axis=0):
    theta = angle * np.pi / 180
    nz, ny, nx = array.shape
    # first translate the middle of the image to the origin
    T1 = np.array(
        [[1, 0, 0, nx / 2], [0, 1, 0, ny / 2], [0, 0, 1, nz / 2], [0, 0, 0, 1]]
    )
    # then rotate theta degrees about the Y axis
    if axis in (0, "z", "Z"):
        R = np.array(
            [
                [np.cos(theta), np.sin(theta), 0, 0],
                [-np.sin(theta), np.cos(theta), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
    elif axis in (1, "y", "Y"):
        R = np.array(
            [
                [np.cos(theta), 0, -np.sin(theta), 0],
                [0, 1, 0, 0],
                [np.sin(theta), 0, np.cos(theta), 0],
                [0, 0, 0, 1],
            ]
        )
    elif axis in (2, "x", "X"):
        R = np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(theta), np.sin(theta), 0],
                [0, -np.sin(theta), np.cos(theta), 0],
                [0, 0, 0, 1],
            ]
        )
    else:
        raise ValueError("Unrecognized axis of rotation: {}".format(axis))
    # then translate back to the original origin
    T2 = np.array(
        [[1, 0, 0, -nx / 2], [0, 1, 0, -ny / 2], [0, 0, 1, -nz / 2], [0, 0, 0, 1]]
    )

    T = np.dot(np.dot(np.dot(np.eye(4), T1), R), T2)
    return T


@with_cupy_conversion
def scale(array, scalar=(1, 1, 1), mode="nearest", blocks=(16, 16, 4)):
    """scale array with nearest neighbors or linear interpolation

    scale can be either a scalar or a tuple with len(scale) == array.ndim
    """
    if isinstance(scalar, (int, float)):
        scalar = (scalar, scalar, scalar)

    _dtype = array.dtype
    if not _dtype == np.float32:
        array = array.astype(np.float32)

    # make scaling array
    tmat = _make_scaling_matrix(scalar)

    # bind array to textureRef
    _bind_tex(array)
    _set_tex_filter_mode(mode)

    out_z = round(array.shape[0] * scalar[0])
    out_y = round(array.shape[1] * scalar[1])
    out_x = round(array.shape[2] * scalar[2])
    output = gpuarray.empty((out_z, out_y, out_x), dtype=np.float32)

    bx, by, bz = blocks
    grid = ((out_x + bx - 1) // bx, (out_y + by - 1) // by, (out_z + bz - 1) // bz)

    _affine(
        cuda.Out(output),
        np.int32(out_x),
        np.int32(out_y),
        np.int32(out_z),
        cuda.In(tmat.astype(np.float32).ravel()),
        texrefs=[texref],
        block=blocks,
        grid=grid,
    )
    return output


@with_cupy_conversion
def rotate(array, angle, axis=0, mode="nearest", blocks=(16, 16, 4)):
    """rotate array around a single axis

    axis can be either 0,1,2 or z,y,x
    """
    _dtype = array.dtype
    if not _dtype == np.float32:
        array = array.astype(np.float32)

    # make scaling array
    tmat = _make_rotation_matrix(array, angle, axis)
    # bind array to textureRef
    _bind_tex(array)
    _set_tex_filter_mode(mode)

    out_z, out_y, out_x = array.shape
    output = gpuarray.empty((out_z, out_y, out_x), dtype=np.float32)
    bx, by, bz = blocks
    grid = ((out_x + bx - 1) // bx, (out_y + by - 1) // by, (out_z + bz - 1) // bz)

    _affine(
        output,
        np.int32(out_x),
        np.int32(out_y),
        np.int32(out_z),
        cuda.In(tmat.astype(np.float32).ravel()),
        texrefs=[texref],
        block=blocks,
        grid=grid,
    )
    return output


@with_cupy_conversion
def translate(array, mag=(0, 0, 0), mode="nearest", blocks=(16, 16, 4)):
    """translate array

    mag is number of pixels to translate in (z,y,x)
    must be tuple with length array.ndim
    """
    _dtype = array.dtype
    if not _dtype == np.float32:
        array = array.astype(np.float32)

    # make scaling array
    tmat = _make_translation_matrix(mag)

    # bind array to textureRef
    _bind_tex(array)
    _set_tex_filter_mode(mode)

    out_z, out_y, out_x = array.shape
    output = gpuarray.empty((out_z, out_y, out_x), dtype=np.float32)
    bx, by, bz = blocks
    grid = ((out_x + bx - 1) // bx, (out_y + by - 1) // by, (out_z + bz - 1) // bz)

    _affine(
        output,
        np.int32(out_x),
        np.int32(out_y),
        np.int32(out_z),
        cuda.In(tmat.astype(np.float32).ravel()),
        texrefs=[texref],
        block=blocks,
        grid=grid,
    )
    return output


@with_cupy_conversion
def affine(array, tmat, mode="nearest", blocks=(16, 16, 4)):
    """translate array

    mag is number of pixels to translate in (z,y,x)
    must be tuple with length array.ndim
    """
    _dtype = array.dtype
    if not _dtype == np.float32:
        array = array.astype(np.float32)

    assert tmat.shape == (4, 4), "transformation matrix must have shape (4, 4)"

    # bind array to textureRef
    _bind_tex(array)
    _set_tex_filter_mode(mode)

    out_z, out_y, out_x = array.shape
    output = gpuarray.empty((out_z, out_y, out_x), dtype=np.float32)
    bx, by, bz = blocks
    grid = ((out_x + bx - 1) // bx, (out_y + by - 1) // by, (out_z + bz - 1) // bz)

    _affine(
        output,
        np.int32(out_x),
        np.int32(out_y),
        np.int32(out_z),
        cuda.In(tmat.astype(np.float32).ravel()),
        texrefs=[texref],
        block=blocks,
        grid=grid,
    )
    return output
