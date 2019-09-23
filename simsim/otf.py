from simsim.illum_pycuda import structillum_3d_tex, structillum_3d
from simsim.psf import psf
import mrc
import numpy as np
from skimage.transform import downscale_local_mean
import tifffile as tf
import matplotlib.pyplot as plt

plt.ion()


def crop_center(img, cropx, cropy):
    z, y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[:, starty : starty + cropy, startx : startx + cropx]


out_dx = 0.08
out_dz = 0.125
out_nxy = 256
upscale_xy = 8
truth_nxy = out_nxy * upscale_xy
truth_nz = 65
truth_dx = out_dx / upscale_xy
truth_dz = out_dz

nimm = 1.515
NA = 1.42
csthick = 0.170
sample_ri = 1.426
gratingDefocus = 0

# print("making psf")
# _psf = psf(
#     nxy=truth_nxy,
#     nz=truth_nz,
#     dz=truth_dz,
#     dxy=truth_dx,
#     wvl=emwave,
#     NA=NA,
#     csthick=csthick,
#     nimm=1.515,
#     sample_ri=sample_ri,
# )
# _psf /= _psf.sum()

_psf = crop_center(
    mrc.imread(
        "/Users/talley/python/simsim/_local/psf_010x125nm_142NA_528nm_1515nimm_146sampRI.dv"
    ),
    truth_nxy,
    truth_nxy,
)

_psf = np.tile(_psf, (5, 1, 1, 1))

illum_contrast = 1
exwave = 0.488
emwave = 0.528
angles = [0]
linespacing = 0.2035
nphases = 5


print("making illum")
nxyillum = 1
illum_shape = (truth_nz, nxyillum, nxyillum)
illum = structillum_3d_tex(
    illum_shape,
    angles,
    nphases,
    linespacing=linespacing,
    dx=truth_dx,
    dz=truth_dz,
    defocus=gratingDefocus,
    NA=NA,
    nimm=nimm,
    wvl=exwave,
)[0]

illum = illum[:, :, nxyillum // 2, nxyillum // 2, np.newaxis, np.newaxis].get()

illum = np.squeeze(np.tile(illum, (1, 1, 1, truth_nxy, truth_nxy)))
out = np.transpose((_psf * illum), (1, 0, 2, 3)).reshape((-1, truth_nxy, truth_nxy))
# out = np.transpose((_psf), (1, 0, 2, 3)).reshape((-1, truth_nxy, truth_nxy))

# out = (_psf * illum)[0].reshape((-1, truth_nxy, truth_nxy))
final = downscale_local_mean(out, (1, upscale_xy, upscale_xy))

mrc.imsave(
    "/Users/talley/Desktop/psf.dv",
    final.astype(np.float32),
    metadata={"dx": out_dx, "dy": out_dx, "dz": out_dz, "wave0": 1000 * emwave},
)
