import numpy as np
from simsim.transform import rotate, shift
from pycuda import cumath, gpuarray


def crop_center(img, cropx, cropy):
    z, y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[:, starty : starty + cropy, startx : startx + cropx]


def efield(kvec, zarr, xarr, dx, dz):
    return cumath.exp(1j * 2 * np.pi * (kvec[0] * xarr * dx + kvec[1] * zarr * dz))


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

    nz, nx = shape
    anglespan = spotratio * 2 * np.arcsin(NA / nimm)
    NA_span = np.sin(anglespan)
    NA_arr = (
        np.arange(-nangles / 2, nangles / 2 + 1, dtype=np.float32) * NA_span / nangles
    )
    kmag = nimm / wvl

    # The contribution to the illum is dependent on theta, since the middle of the circle
    # has more rays than the edge
    # kmag*cumath.sin(anglespan/2)) is the radius of each circular illumination spot
    # weight_arr is essentially the "chord" length as a function of theta_arr
    _t = kmag * NA_span
    weight_arr = np.sqrt((_t / 2) ** 2 - (kmag * NA_arr) ** 2) / (_t / 2)

    plus_sideNA_arr = (0.5 / linespacing + kmag * NA_arr) / kmag
    minus_sideNA_arr = -plus_sideNA_arr[::-1]

    # intensity = gpuarray.zeros((3, nz + extraz, nx), np.float32)
    intensity = gpuarray.zeros((nz + extraz, nx), np.float32)

    amp = gpuarray.zeros((6, nz + extraz, nx), np.complex64)
    zarr, xarr = np.indices((nz + extraz, nx), np.float32)
    zarr -= (nz + extraz) / 2
    xarr -= nx / 2
    amp_plus = np.sqrt(1.0 - side_intensity).astype(np.complex64)

    zarr = gpuarray.to_gpu(zarr)
    xarr = gpuarray.to_gpu(xarr)
    kvec_arr = kmag * np.stack([NA_arr, np.sqrt(1 - NA_arr ** 2)]).transpose()
    kvec_arr_plus = (
        kmag
        * np.stack([plus_sideNA_arr, np.sqrt(1 - plus_sideNA_arr ** 2)]).transpose()
    )
    kvec_arr_minus = (
        kmag
        * np.stack([minus_sideNA_arr, np.sqrt(1 - minus_sideNA_arr ** 2)]).transpose()
    )

    ampcenter = np.complex64(ampcenter)
    for i, wght in enumerate(weight_arr):
        # construct intensity field over all triplets

        amp[0] = amp_plus * efield(kvec_arr[i], zarr, xarr, dx, dz) * ampcenter
        amp[2] = amp_plus * efield(kvec_arr_plus[i], zarr, xarr, dx, dz) * ampratio
        amp[4] = amp_plus * efield(kvec_arr_minus[i], zarr, xarr, dx, dz) * ampratio

        intensity += (
            (amp[0] * amp[0].conj() + amp[2] * amp[2].conj() + amp[4] * amp[4].conj())
            * wght
        ).real
        intensity += (
            2 * (amp[0] * amp[2].conj() + amp[0] * amp[4].conj()).real * wght
        )
        intensity += 2 * (amp[2] * amp[4].conj()).real * wght

    del amp

    if extraz > 0:
        aslope = np.arange(extraz, dtype=np.float32) / extraz
        blend = np.transpose(
            np.transpose(intensity[:extraz, :]) * aslope
            + np.transpose(intensity[-extraz:, :]) * (1 - aslope)
        )
        intensity[:extraz, :] = blend
        intensity[-extraz:, :] = blend
        return intensity[extraz // 2 : -extraz // 2, :]
    return intensity


def structillum_3d(
    shape,
    angles=0,
    nphases=5,
    linespacing=0.2035,
    dx=0.01,
    dz=0.01,
    defocus=0,
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
    phaseshift = 2 * linespacing / nphases
    kwargs["linespacing"] = linespacing
    kwargs["dz"] = dz
    kwargs["dx"] = dx
    nz, ny, nx = shape

    # adding a single pixel to z and removing to make focal plane centered
    shape_2d = (shape[0] + 1, int(np.ceil(shape[1] * 1.55)))
    ill_2d = structillum_2d(shape_2d, *args, **kwargs)[:-1]
    # ill_3d = xp.repeat(ill_2d[:, :, xp.newaxis], np.int(shape[2] * np.sqrt(2)), axis=2)

    # ndimage.rotate(ill_3d, 45, (1, 2))

    out = gpuarray.zeros((nangles, nphases, *shape), dtype=np.float32)  # APZYX shape
    for p in range(nphases):
        shiftedIllum = shift(ill_2d, (defocus / dz, p * phaseshift / dx)).get()
        ill_3d = np.repeat(
            shiftedIllum[:, :, np.newaxis], np.ceil(shape[2] * np.sqrt(2)), axis=2
        )
        for a, angle in enumerate(angles):
            print(f"p: {p}, a: {a}")
            if angle == 0:
                rotatedillum = ill_3d
            else:
                rotatedillum = rotate(ill_3d, np.rad2deg(angle), mode="linear")
            out[a, p] = crop_center(rotatedillum, nx, ny)
    return out

    # out = np.empty((nangles, nphases, *shape), xp.float32)  # APZYX shape
    # tempy = np.int(shape[2] * np.sqrt(2))
    # for p in range(nphases):
    #     shifted = shift(ill_2d, (defocus / dz, p * phaseshift / dx), order=1)
    #     ill_3d = xp.repeat(shifted[:, :, xp.newaxis], tempy, axis=2).get()
    #     for a, angle in enumerate(angles):
    #         print(f"p: {p}, a: {a}")
    #         if angle == 0:
    #             rotatedillum = ill_3d
    #         else:
    #             rotatedillum = rotate(ill_3d, np.rad2deg(angle), mode="linear").get()
    #         out[a, p] = crop_center(rotatedillum, nx, ny)
    # return out
