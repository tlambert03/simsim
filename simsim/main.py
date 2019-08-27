from simsim.truth.matslines import matslines3D
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
from simsim.illum import structillum_3d
from simsim.illumx import structillum_3d as structillum_3d_gpu
from simsim.psf import psf
import cupy as cp
from cupyx.scipy.ndimage import zoom

plt.ion()


def main():

    nimm = 1.515
    NA = 1.42
    csthick = 0.170
    sample_ri = 1.515
    gratingDefocus = 0

    illum_contrast = 1
    exwave = 0.488
    emwave = 0.528
    angles = [0, np.deg2rad(60), np.deg2rad(120)]
    linespacing = 0.2035
    nphases = 5

    out_nx = 128
    out_ny = 128
    out_nz = 13
    out_dx = 0.08
    out_dz = 0.125
    upscale_xy = 8
    upscale_z = 5
    assert out_nz % 2 == 1 and upscale_z % 2 == 1, "out_nz and upscale_z must be odd"
    truth_nx = out_nx * upscale_xy
    truth_ny = out_ny * upscale_xy
    truth_nz = out_nz * upscale_z
    truth_dx = out_dx / upscale_xy
    truth_dz = out_dz / upscale_z

    # ex
    with cp.cuda.device.Device(0):
        illum_shape = (truth_nz + out_nz // 2 * upscale_z * 2, truth_ny, truth_nx)
        illum = structillum_3d_gpu(
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
        )
        illum = cp.array(illum)
    # normalize
    illum -= illum.min()
    illum *= 1 / illum.max()
    # adjust contrast
    illum *= max(0, min(illum_contrast, 1))
    illum += 1 - illum.max()

    truth = matslines3D((truth_nz, truth_ny, truth_nx), density=5).astype(np.float32)

    _psf = psf(
        nxy=truth_nx,
        nz=truth_nz,
        dz=truth_dz,
        dxy=truth_dx,
        wvl=emwave,
        NA=NA,
        csthick=csthick,
        nimm=1.515,
        sample_ri=sample_ri,
    )
    _psf /= _psf.sum()

    try:
        del res_gpu  # noqa
    except NameError:
        pass
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()
    out = np.empty((len(angles), nphases, out_nz, out_ny, out_nx), np.float32)
    truth_gpu = cp.asarray(truth)
    otf_gpu = cp.fft.fftn(cp.asarray(_psf))
    for plane in range(out_nz):
        start = plane * upscale_z
        need_plane = truth_nz - 1 - (upscale_z // 2 + (plane * upscale_z))
        print(f"illum_seg: {start} to {start + truth_nz}")
        print(f"extracting plane {need_plane}")
        for angle in range(len(angles)):
            for phase in range(nphases):
                print(f"plane: {plane}, angle: {angle}, phase: {phase}")
                res_gpu = cp.fft.ifftn(
                    cp.fft.fftn(
                        cp.multiply(illum[angle, phase, start : start + truth_nz], truth_gpu)
                    )
                    * otf_gpu
                ).real
                res_gpu = cp.fft.fftshift(res_gpu)[need_plane].astype(np.float32)
                out[angle, phase, plane] = zoom(
                    res_gpu, (1 / upscale_xy, 1 / upscale_xy)
                ).get()
                del res_gpu
                mempool.free_all_blocks()

    # APZ -> PZA
    _out = np.transpose(out, (0, 2, 1, 3, 4)).reshape((-1, out.shape[3], out.shape[4]))
    #    illum_gpu = cp.asarray(illum)


if __name__ == "__main__":
    main()
