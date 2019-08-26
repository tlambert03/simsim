import numpy as np


def gauss1d(mean=0, sigma=1, amp=1):
    def func(x):
        return amp * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))

    return func


def gauss2d(mean=(0, 0), sigma=(1, 1), amp=1):
    def func(x, y):
        sigx, sigy = sigma
        meanx, meany = mean
        return amp * np.exp(
            -((x - meanx) ** 2 / (2 * sigx ** 2) + (y - meany) ** 2 / (2 * sigy ** 2))
        )

    return func


def gauss3d(mean=(0, 0, 0), sigma=(1, 1, 1), amp=1):
    def func(x, y, z):
        sigx, sigy, sigz = sigma
        meanx, meany, meanz = mean
        return amp * np.exp(
            -(
                (x - meanx) ** 2 / (2 * sigx ** 2)
                + (y - meany) ** 2 / (2 * sigy ** 2)
                + (z - meanz) ** 2 / (2 * sigz ** 2)
            )
        )

    return func


def gauss3d_kernel(shape=(100, 128, 128), sigx=1, sigz=2):
    f = gauss3d(mean=(0, 0, 0), sigma=(sigx, sigx, sigz))
    z, y, x = np.mgrid[
        -shape[0] // 2 : shape[0] // 2,
        -shape[1] // 2 : shape[1] // 2,
        -shape[2] // 2 : shape[2] // 2,
    ]
    a = f(x, y, z)
    return a


def gauss_psf(nxy=128, nz=128, dz=0.1625, dxy=0.1625, wvl=0.55, NA=0.8):
    sigx = ((0.61 * wvl / NA) / 2.355) / dxy
    sigz = ((2 * wvl / NA ** 2) / 2.355) / dz
    psf = gauss3d_kernel((nz, nxy, nxy), sigx, sigz)
    return psf
