import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from tqdm import tqdm


try:
    import cupy as cp

    xp = cp
    from cupy.cuda.memory import OutOfMemoryError
    from cupyx.scipy.ndimage import map_coordinates as cu_map_coordinates
except ImportError:
    cp = None
    xp = np

try:
    import cupy.cuda.texture as cptex
except ImportError:
    print("cupy.cuda has no texture module")
    cptex = None

_DEFAULT_LS = 0.2035


def efield(kvec, zarr, xarr, dx, dz):
    field = np.exp(1j * 2 * np.pi * (kvec[0] * xarr * dx + kvec[1] * zarr * dz))
    return field


def structillum_2d(
    shape,
    dx=0.01,
    dz=0.01,
    NA=1.42,
    nimm=1.515,
    wvl=0.488,
    linespacing=0.2035,
    extraz=0,
    side_intensity=0.5,
    ampcenter=1.0,
    ampratio=1.0,
    nangles=100,
    spotratio=0.035,
):
    """ Simulate a plane of structured illumination intensity either for 1 or 2 objectives
    '2d' means I'm only creating one sheet of illumination since every sheet will be the
    same side_intensity (0~1) -- the amplitude of illum from one objective;
        for the other it's 1 minus this value
    ampcenter -- the amplitude of the center illum beam;
        if 0 and OneObj, then it's for 2D SIM
    ampratio -- the amplitude of the side beams relative to center beam, which is 1.0
    nangles -- the number of triplets (or sextets) we'd divide the illumination beams
        into because the beams assume different incident angles (multi-mode fiber)
    """

    # theta_arr = np.arange(-nangles/2, nangles/2+1, dtype=np.float32) * anglespan/nangles

    nz, nx = shape

    # I think NA is half angle, therefore 2*
    anglespan = spotratio * 2 * np.arcsin(NA / nimm)

    NA_span = np.sin(anglespan)
    NA_arr = (
        np.arange(-nangles / 2, nangles / 2 + 1, dtype=np.float32) * NA_span / nangles
    )

    kmag = nimm / wvl

    # The contribution to the illum is dependent on theta, since the middle of the circle
    # has more rays than the edge
    # kmag*np.sin(anglespan/2)) is the radius of each circular illumination spot
    # weight_arr is essentially the "chord" length as a function of theta_arr
    # weight_arr = np.sqrt(
    #   (kmag*np.sin(anglespan/2)) ** 2 - (kmag*np.sin(theta_arr))**2 )
    #   / (kmag*np.sin(anglespan/2))
    weight_arr = np.sqrt((kmag * NA_span / 2) ** 2 - (kmag * NA_arr) ** 2) / (
        kmag * NA_span / 2
    )

    # plus_sidetheta_arr  = np.arcsin( (kmag * np.sin(theta_arr) + 1/linespacing/2)/kmag )
    # minus_sidetheta_arr = -plus_sidetheta_arr[::-1]

    plus_sideNA = (1 / linespacing / 2 + kmag * NA_arr) / kmag
    minus_sideNA = -plus_sideNA[::-1]

    #     intensity = np.zeros((nz+extraz,nx), np.float32)
    intensity = np.zeros((3, nz + extraz, nx), np.float32)

    amp = np.zeros((3, nz + extraz, nx), np.complex64)
    zarr, xarr = np.indices((nz + extraz, nx)).astype(np.float32)
    zarr -= (nz + extraz) / 2
    xarr -= nx / 2

    amp_plus = np.sqrt(1.0 - side_intensity)
    # amp_minus = np.sqrt(side_intensity)

    kvecs = kmag * np.stack([NA_arr, np.sqrt(1 - NA_arr ** 2)]).T
    plus_kvecs = kmag * np.stack([plus_sideNA, np.sqrt(1 - plus_sideNA ** 2)]).T
    minus_kvecs = kmag * np.stack([minus_sideNA, np.sqrt(1 - minus_sideNA ** 2)]).T

    for i, wght in enumerate(weight_arr):
        amp[0] = amp_plus * efield(kvecs[i], zarr, xarr, dx, dz) * ampcenter
        amp[1] = amp_plus * efield(plus_kvecs[i], zarr, xarr, dx, dz) * ampratio
        amp[2] = amp_plus * efield(minus_kvecs[i], zarr, xarr, dx, dz) * ampratio

        intensity[0] += (
            (amp[0] * amp[0].conj() + amp[1] * amp[1].conj() + amp[2] * amp[2].conj())
            * wght
        ).real
        intensity[1] += (
            2 * np.real(amp[0] * amp[1].conj() + amp[0] * amp[2].conj()) * wght
        )
        intensity[2] += 2 * np.real(amp[1] * amp[2].conj()) * wght

    del amp

    if extraz > 0:
        # blend = F.zeroArrF(extraz, nx)
        aslope = np.arange(extraz, dtype=np.float32) / extraz
        blend = np.transpose(
            np.transpose(intensity[:extraz, :]) * aslope
            + np.transpose(intensity[-extraz:, :]) * (1 - aslope)
        )
        intensity[:extraz, :] = blend
        intensity[-extraz:, :] = blend
        return intensity[extraz // 2 : -extraz // 2, :]
    else:
        return intensity


def _single_period(nz, *args, resolution=100, **kwargs):
    # returns a single period of lower frequency of the the 2D pattern
    # Z extent is unchanged, but pixel size is changed to be linespacing/resolution
    # and returned as the second value in the return tuple
    kwargs["linespacing"] = kwargs.get("linespacing", _DEFAULT_LS)
    kwargs["dx"] = kwargs["linespacing"] / resolution
    return structillum_2d((nz, 2 * resolution), *args, **kwargs), kwargs["dx"]


def structillum_3d(
    shape,
    angles=0,
    nphases=5,
    linespacing=0.2035,
    dx=0.01,
    dz=0.01,
    defocus=0,
    xp=xp,
    *args,
    **kwargs,
):
    if isinstance(angles, (int, float)):
        # if a single number is provided, assume it is the first of three
        angles = [angles, angles + np.deg2rad(60), angles + np.deg2rad(120)]
    assert isinstance(
        angles, (list, tuple)
    ), "Angles argument should be a list of angles in radians"
    nangles = len(angles)
    phaseshift = 2 * linespacing / nphases / dx

    nz, ny, nx = shape
    kwargs["linespacing"] = linespacing
    kwargs["dz"] = dz
    per, per_dxy = _single_period(nz + 1, resolution=100, **kwargs)
    per = xp.asarray(per.sum(0)[1:]).T
    # out = xp.empty((nangles, nphases, ny * nz * nx), dtype=xp.float32)
    out = np.empty((nangles, nphases, nz, ny, nx), dtype=xp.float32)

    coords = xp.indices((ny, nz, nx)).reshape((3, -1))

    _scale = xp.eye(4)
    _scale[0, 0] = dx / per_dxy
    _scale[2, 2] = 0  # flatten the z dimension to the 2D plane

    with tqdm(total=(nangles * nphases * 2)) as pbar:
        for a, theta in enumerate(angles):
            for p in range(nphases):
                pbar.set_description(
                    f"angle {a + 1}/{nangles}, phase {p + 1}/{nphases}"
                )
                _sin = np.sin(theta)
                _cos = np.cos(theta)
                _rot = xp.array(
                    [
                        [_cos, 0, -_sin, 0],
                        [0, 1, 0, 0],
                        [_sin, 0, _cos, 0],
                        [0, 0, 0, 1],
                    ]
                )
                # translate
                _shift = xp.eye(4)
                _shift[0, 3] = p * phaseshift
                matrix = xp.dot(xp.dot(_scale, _shift), _rot)
                offset = matrix[:-1, -1]
                matrix = matrix[:-1, :-1]

                coordinates = xp.dot(matrix, coords)[:2]
                if any(offset):
                    coordinates += xp.expand_dims(xp.asarray(offset[:2]), -1)
                coordinates[0] = coordinates[0] % (per.shape[0] - 1)
                pbar.update(1)
                if xp.__name__ == "cupy":
                    out[a, p] = (
                        cu_map_coordinates(per, coordinates, out.dtype, order=1)
                        .reshape((ny, nz, nx))
                        .transpose((1, 2, 0))
                        .get()
                    )
                else:
                    out[a, p] = (
                        map_coordinates(per, coordinates, out.dtype, order=1)
                        .reshape((ny, nz, nx))
                        .transpose((1, 2, 0))
                    )
                pbar.update(1)
    return out


source = r"""
extern "C"{
    __global__ void readTex(float *output, cudaTextureObject_t texObj, float *mat,
                            int nx, int ny, int nz, int texwidth) {
        // Calculate oputput coordinates
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
        if (x >= nx || y >= ny || z >= nz) {
            return;
        }

        float u = x;
        float v = z;
        float w = y;

        float tu = mat[0] * u + mat[1] * v + mat[2] * w + mat[3] + 0.5f;
        float tv = mat[4] * u + mat[5] * v + mat[6] * w + mat[7] + 0.5f;

        // Read from texture and write to global memory
        int idx = z * (nx * ny) + y * nx + x;

        // Since wrap mode does not seem to work
        if (tu <= 0 || tu >= texwidth) {
            tu = abs(fmod(tu, texwidth));
        }

        output[idx] = tex2D<float>(texObj, tu, tv);
    }
}
"""


def make_tex_ob(tex_data):
    # set up a texture object
    height, width = tex_data.shape
    ch = cptex.ChannelFormatDescriptor(
        32, 0, 0, 0, cp.cuda.runtime.cudaChannelFormatKindFloat
    )
    arr2 = cptex.CUDAarray(ch, width, height)
    res = cptex.ResourceDescriptor(cp.cuda.runtime.cudaResourceTypeArray, cuArr=arr2)
    tex = cptex.TextureDescriptor(
        (cp.cuda.runtime.cudaAddressModeWrap, cp.cuda.runtime.cudaAddressModeWrap),
        cp.cuda.runtime.cudaFilterModeLinear,
        cp.cuda.runtime.cudaReadModeElementType,
    )
    texobj = cptex.TextureObject(res, tex)
    arr2.copy_from(tex_data)
    return texobj


def _make_grid(shape, blocks):
    if len(shape) == 3:
        out_z, out_y, out_x = shape
        bx, by, bz = blocks
        return ((out_x + bx - 1) // bx, (out_y + by - 1) // by, (out_z + bz - 1) // bz)
    elif len(shape) == 2:
        out_y, out_x = shape
        bx, by = blocks[:2]
        return ((out_x + bx - 1) // bx, (out_y + by - 1) // by, 1)


def read_coords(shape, tex_data, matrix, blocks=(16, 16, 4), output=None):
    nz, ny, nx = shape
    texobj = make_tex_ob(tex_data.astype("float32"))
    grid = _make_grid(shape, blocks)
    ker = cp.RawKernel(source, "readTex")
    if output is None:
        output = cp.zeros(shape, dtype="float32")
    ker(
        grid,
        blocks,
        (
            output,
            texobj,
            cp.asarray(matrix.ravel(), dtype="float32"),
            nx,
            ny,
            nz,
            texobj.ResDesc.cuArr.width,
        ),
    )
    return output


def structillum_3d_tex(
    shape,
    angles=0,
    nphases=5,
    linespacing=0.2035,
    dx=0.01,
    dz=0.01,
    defocus=0,
    transfer=True,  # transfer it back to host
    *args,
    **kwargs,
):
    if not cptex:
        raise ImportError(
            "Cannot run this version of structillum without cupy.cuda.texture. "
            "Please update to a newer version of cupy"
        )
    if isinstance(angles, (int, float)):
        # if a single number is provided, assume it is the first of three
        angles = [angles, angles + np.deg2rad(60), angles + np.deg2rad(120)]
    assert isinstance(
        angles, (list, tuple)
    ), "Angles argument should be a list of angles in radians"
    nangles = len(angles)
    phaseshift = 2 * linespacing / nphases / dx
    nz, ny, nx = shape
    kwargs["linespacing"] = linespacing
    kwargs["dz"] = dz
    per, per_dxy = _single_period(nz + 1, resolution=100, **kwargs)
    per = per.sum(0)[1:]

    if transfer:
        out = np.empty((nangles, nphases, nz, ny, nx), dtype=xp.float32)
    else:
        out = cp.empty((nangles, nphases, nz, ny, nx), dtype=xp.float32)

    _scale = xp.eye(4)
    _scale[0, 0] = dx / per_dxy
    _scale[2, 2] = 0  # flatten the z dimension to the 2D plane

    with tqdm(total=(nangles * nphases)) as pbar:
        for a, theta in enumerate(angles):
            for p in range(nphases):
                pbar.set_description(
                    f"angle {a + 1}/{nangles}, phase {p + 1}/{nphases}"
                )
                _sin = np.sin(theta)
                _cos = np.cos(theta)
                _rot = xp.array(
                    [
                        [_cos, 0, -_sin, 0],
                        [0, 1, 0, 0],
                        [_sin, 0, _cos, 0],
                        [0, 0, 0, 1],
                    ]
                )
                _shift = xp.eye(4)
                _shift[0, 3] = p * phaseshift
                matrix = _scale @ _shift @ _rot
                if transfer:
                    out[a, p] = read_coords(shape, per, matrix[:2]).get()
                else:
                    read_coords(shape, per, matrix[:2], output=out[a, p])
                pbar.update(1)
    return out


def structillum_3d_with_fallback(*args, **kwargs):
    try:
        return structillum_3d_tex(*args, transfer=True, **kwargs)
    except ImportError:
        return structillum_3d(*args, **kwargs)
    except OutOfMemoryError:
        print("out of gpu memory falling back to cpu")
        kwargs["xp"] = np
        return structillum_3d(*args, **kwargs)


structillum_3d_with_fallback.__doc__ = structillum_3d.__doc__

