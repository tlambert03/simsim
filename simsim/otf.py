from simsim.psf import PSF
from simsim.illum import SIMIllum

import numpy as np
from skimage.transform import downscale_local_mean


def make_otf(
    shape=(65, 256, 256),
    wvl=0.525,
    dx=0.08,
    dz=0.125,
    upscale_x=1,
    nimm=1.515,
    NA=1.42,
    pattern_defocus=0,
    modulation_contrast=1,
    sample_ri=1.515,
    pz=0,
    angle=0,
    nphases=5,
    linespacing=0.2305,
    outpath=None,
):
    truth_nz, out_nxy, out_nxy = shape
    truth_nxy = out_nxy * upscale_x
    truth_dx = dx / upscale_x
    truth_dz = dz

    params = {
        "NA": NA,  # numerical aperture
        "ng0": 1.515,  # coverslip RI design value
        "ng": 1.515,  # coverslip RI experimental value
        "ni0": 1.515,  # immersion medium RI design value
        "ni": nimm,  # immersion medium RI experimental value
        "ns": sample_ri,  # specimen refractive index (RI)
        "ti0": 150,  # microns, working distance (immersion medium thickness) design value
        "tg": 170,  # microns, coverslip thickness experimental value
        "tg0": 170,  # microns, coverslip thickness design value
    }

    psf = PSF(shape, params, truth_dx, truth_dz, pz, wvl)
    illum = SIMIllum(
        (truth_nz, 1, 1),
        truth_dx,
        truth_dz,
        [angle],
        nphases,
        linespacing,
        pattern_defocus,
        modulation_contrast,
        NA,
        nimm,
        wvl,
    )
    otf = psf.data[np.newaxis, np.newaxis] * illum.data
    otf = otf.transpose((0, 2, 1, 4, 3)).reshape((-1, truth_nxy, truth_nxy))
    otf = downscale_local_mean(otf, (1, upscale_x, upscale_x)).astype("float32")
    if outpath:
        import mrc

        mrc.imsave(
            outpath,
            otf,
            metadata={
                "dx": dx,
                "dy": dx,
                "dz": dz,
                "wave0": 1000 * wvl,
                "LensNum": 10612,
            },
        )
    return otf
