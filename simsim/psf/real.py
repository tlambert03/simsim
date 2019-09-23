import microscPSF.microscPSF as mpsf
import numpy as np


def real_psf(
    nxy=128,
    nz=128,
    dz=0.1625,
    dxy=0.1625,
    wvl=0.55,
    NA=1.42,
    mag=100,
    nimm=1.515,
    sample_ri=1.42,
    csthick=0.170,
    ti0=150,
):
    params = {
        "M": mag,  # magnification
        "NA": NA,  # numerical aperture
        "ng0": 1.515,  # coverslip RI design value
        "ng": 1.515,  # coverslip RI experimental value
        "ni0": 1.515,  # immersion medium RI design value
        "ni": nimm,  # immersion medium RI experimental value
        "ns": sample_ri,  # specimen refractive index (RI)
        "ti0": ti0,  # microns, working distance (immersion medium thickness) design value
        "tg": csthick * 1000,  # microns, coverslip thickness experimental value
        "tg0": 170,  # microns, coverslip thickness design value
        "zd0": 200.0 * 1.0e3,  # microscope tube length (in microns).
    }
    zv = np.arange(-(nz - 1) // 2, (nz + 1) // 2) * dz
    psf = mpsf.gLXYZFocalScan(
        params, dxy, nxy, zv, normalize=True, pz=0.0, wvl=wvl, zd=None
    )
    return psf


def psf(real=True, **kwargs):
    if real:
        try:
            return real_psf(**kwargs)
        except ImportError:
            print("could not import microscPSF, falling back to 3D gaussian PSF")
            pass
    return gauss_psf(**kwargs)


def spim_psf(sheet_fwhm=3, real=True, **kwargs):
    _psf = psf(real, **kwargs)
    sheet_sig = (sheet_fwhm / 2.355) / kwargs.get("dz", 0.1625)
    z = np.arange(-_psf.shape[0] // 2, _psf.shape[0] // 2)
    sheet_profile = gauss1d(z, sheet_sig)
    newpsf = sheet_profile[:, np.newaxis, np.newaxis] * _psf
    return newpsf / newpsf.sum()
