import numpy as np
from scipy.ndimage import shift


def expfield_2d(kvec, zarr, xarr, dx, dz):
    field = np.exp(1j * 2 * np.pi * (kvec[0] * xarr * dx + kvec[1] * zarr * dz))
    return field


def _single_period(*args, **kwargs):
    resolution = 100
    _dz = kwargs.get("dz", 0.01)
    kwargs["linespacing"] = kwargs.get("linespacing", 0.2035)
    kwargs["dx"] = kwargs["linespacing"] / resolution
    kwargs["dz"] = kwargs["dx"]
    args = list(args)
    args[0] = (int(args[0][0] * _dz / kwargs["dz"]), 2 * resolution)
    return structillum_2d(*args, **kwargs), kwargs["dx"]


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

    plus_sideNA_arr = (1 / linespacing / 2 + kmag * NA_arr) / kmag
    minus_sideNA_arr = -plus_sideNA_arr[::-1]

    #     intensity = np.zeros((nz+extraz,nx), np.float32)
    intensity = np.zeros((3, nz + extraz, nx), np.float32)

    amplitude = np.zeros((6, nz + extraz, nx), np.complex64)
    zarr, xarr = np.indices((nz + extraz, nx)).astype(np.float32)
    zarr -= (nz + extraz) / 2
    xarr -= nx / 2

    amp_plus = np.sqrt(1.0 - side_intensity)
    # amp_minus = np.sqrt(side_intensity)

    kvec_arr = kmag * np.stack([NA_arr, np.sqrt(1 - NA_arr ** 2)]).transpose()
    plus_side_kvec_arr = (
        kmag
        * np.stack([plus_sideNA_arr, np.sqrt(1 - plus_sideNA_arr ** 2)]).transpose()
    )
    minus_side_kvec_arr = (
        kmag
        * np.stack([minus_sideNA_arr, np.sqrt(1 - minus_sideNA_arr ** 2)]).transpose()
    )

    for i in range(nangles + 1):
        # construct intensity field over all triplets (or sextets in I5S)
        # print "i=",i
        # amplitude[:]=0j
        #         kvec = kmag * np.stack([np.sin(theta_arr[i]), np.cos(theta_arr[i])])
        # amplitude += expfield_2d(kvec, zarr, xarr, dx, dz)
        amplitude[0] = (
            amp_plus * expfield_2d(kvec_arr[i], zarr, xarr, dx, dz) * ampcenter
        )

        # kvec = kmag * np.stack([np.sin(plus_sidetheta_arr[i]),
        #                        np.cos(plus_sidetheta_arr[i])])
        # amplitude += expfield_2d(kvec, zarr, xarr, dx, dz) * ampratio
        amplitude[2] = (
            amp_plus * expfield_2d(plus_side_kvec_arr[i], zarr, xarr, dx, dz) * ampratio
        )
        # kvec = kmag * np.array([np.sin(minus_sidetheta_arr[i]),
        #                         np.cos(minus_sidetheta_arr[i])])
        # amplitude += expfield_2d(kvec, zarr, xarr, dx, dz) * ampratio
        amplitude[4] = (
            amp_plus
            * expfield_2d(minus_side_kvec_arr[i], zarr, xarr, dx, dz)
            * ampratio
        )
        # intensity += np.absolute(np.sum(amplitude, 0) * weight_arr[i]) ** 2
        intensity[0] += (
            (
                amplitude[0] * amplitude[0].conj()
                + amplitude[2] * amplitude[2].conj()
                + amplitude[4] * amplitude[4].conj()
            )
            * weight_arr[i]
        ).real
        intensity[1] += (
            2
            * np.real(
                amplitude[0] * amplitude[2].conj() + amplitude[0] * amplitude[4].conj()
            )
            * weight_arr[i]
        )
        intensity[2] += 2 * np.real(amplitude[2] * amplitude[4].conj()) * weight_arr[i]

    del amplitude

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


def crop_center(img, cropx, cropy):
    z, y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[:, starty : starty + cropy, startx : startx + cropx]


def structillum_3d(
    shape,
    angles=None,
    nphases=5,
    linespacing=0.2035,
    dx=0.01,
    dz=0.01,
    defocus=0,
    *args,
    **kwargs,
):
    from .cuda.transform import rotate

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
    _shape = (shape[0] + 1, int(np.ceil(shape[1] * 1.55)))
    ill_2d = structillum_2d(_shape, *args, **kwargs).sum(0)[:-1]

    out = np.zeros((nangles, nphases, *shape), "single")  # APZYX shape
    for p in range(nphases):
        shiftedIllum = shift(ill_2d, (defocus / dz, p * phaseshift / dx))
        ill_3d = np.repeat(
            shiftedIllum[:, :, np.newaxis], np.ceil(shape[2] * np.sqrt(2)), axis=2
        )

        for a, angle in enumerate(angles):
            print(f"p: {p}, a: {a}")
            if angle == 0:
                rotatedillum = ill_3d
            else:
                rotatedillum = rotate(ill_3d, np.rad2deg(angle), mode="linear").get()
            out[a, p] = crop_center(rotatedillum, nx, ny)
    return out


if __name__ == "__main__":
    i = structillum_2d((500, 2200))

    import tifffile as tf
    import matplotlib.pyplot as plt

    tf.imshow(i, photometric="minisblack")
    plt.show()
    tf.imshow(i.sum(0))
    plt.show()
