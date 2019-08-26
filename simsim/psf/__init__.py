from .real import real_psf
from .gauss import gauss_psf, gauss1d
import numpy as np


def psf(gauss=False, sheet_fwhm=None, **kwargs):
    _psf = gauss_psf(**kwargs) if gauss else real_psf(**kwargs)
    if sheet_fwhm and sheet_fwhm > 0:
        sheet_sig = (sheet_fwhm / 2.355) / kwargs.get("dz", 0.1625)
        z = np.arange(-_psf.shape[0] // 2, _psf.shape[0] // 2)
        sheet_profile = gauss1d(z, sheet_sig)
        _psf = sheet_profile[:, np.newaxis, np.newaxis] * _psf
        _psf /= _psf.sum()
    return _psf
