# from simsim.cuda import initpycuda
# import simsim.cuda.initpycuda
from pycuda.autoinit import context
from simsim.truth import matslines
import numpy as np
from simsim.illum_pycuda import structillum_3d_tex
from psfmodels import vectorial_psf_centered as vpsf
from pycuda import gpuarray
import pycuda.driver as cuda
from reikna import fft
from reikna.cluda.cuda import Thread
from simsim.transform import zoom


free_mem = cuda.mem_get_info()[0]


def crop_center(img, cropx, cropy):
    z, y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[:, starty : starty + cropy, startx : startx + cropx]


def check_mem():
    global free_mem
    newmem = cuda.mem_get_info()
    print(f"{100 * np.divide(*newmem):0.2f}% free")
    print(f"{(free_mem - newmem[0])/1000000000} used since last time")
    free_mem = newmem[0]


def main():

    out_nx = 128
    out_nz = 13
    upscale_xy = 8
    upscale_z = 5
    out_ny = out_nx

    nimm = 1.515
    NA = 1.40
    csthick = 0.170
    sample_ri = 1.42
    gratingDefocus = 0

    illum_contrast = 1
    exwave = 0.488
    emwave = 0.528
    angles = [-0.8043, -1.855500, 0.238800]
    linespacing = 0.2035
    nphases = 5

    out_dx = 0.08
    out_dz = 0.125
    assert out_nz % 2 == 1 and upscale_z % 2 == 1, "out_nz and upscale_z must be odd"
    truth_nx = out_nx * upscale_xy
    truth_ny = out_ny * upscale_xy
    truth_nz = out_nz * upscale_z
    truth_dx = out_dx / upscale_xy
    truth_dz = out_dz / upscale_z

    # check_mem()

    print("making truth")
    truth = matslines.matslines3D((truth_nz, truth_ny, truth_nx), density=2).astype(
        np.float32
    )
    print("done with truth")
    # check_mem()
    print("making illum")
    illum_shape = (truth_nz + out_nz // 2 * upscale_z * 2, truth_ny, truth_nx)
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
    )
    illum = np.ascontiguousarray(illum.get())

    # check_mem()
    print("norm illum")
    if isinstance(illum, gpuarray.GPUArray):
        # normalize
        illum -= gpuarray.min(illum).get()
        illum *= 1 / gpuarray.max(illum).get()
        # adjust contrast
        illum *= max(0, min(illum_contrast, 1))
        illum += 1 - gpuarray.max(illum).get()
    else:
        illum -= illum.min()
        illum *= 1 / illum.max()
        # adjust contrast
        illum *= max(0, min(illum_contrast, 1))
        illum += 1 - illum.max()

    trimz = -(illum.shape[0] % 2 - 1)
    trimx = -(illum.shape[-1] % 2 - 1)
    psf_nz = illum.shape[0] - trimz
    psf_nx = illum.shape[-1] - trimx
    psf = vpsf(
        psf_nz,
        dz=truth_dz,
        nx=psf_nx,
        dxy=truth_dx,
        pz=0,
        wvl=emwave,
        params={
            'NA': NA,
            "ni0": 1.515,
            "ni": nimm,
            "ng0": 1.515,
            "ng": 1.515,
            "ns": sample_ri,
            "tg0": 170,
            "tg": csthick,
            "ti0": 150,
        }
    )
    psf /= psf.sum()
    psf = np.pad(psf, ((trimz, 0), (trimx, 0), (trimx, 0)))


    # check_mem()
    print("starting conv illum")
    thr = Thread(context)
    print("thread created")
    # check_mem()

    out = np.empty((len(angles), nphases, out_nz, out_ny, out_nx), np.float32)

    truth_gpu = gpuarray.to_gpu(truth)
    print("truth transferred ")
    # check_mem()
    otf_gpu = gpuarray.to_gpu(psf.astype(np.complex64))
    print("otf transferred ")
    # check_mem()
    do_fft = fft.FFT(otf_gpu).compile(thr, fast_math=True)
    do_fft_shift = fft.FFTShift(otf_gpu).compile(thr, fast_math=True)
    print("plans created ")
    # check_mem()
    do_fft(otf_gpu, otf_gpu, inverse=0)
    print("otf fft performed ")
    # check_mem()
    for plane in range(out_nz):

        start = plane * upscale_z
        need_plane = truth_nz - 1 - (upscale_z // 2 + (plane * upscale_z))
        # print(f"illum_seg: {start} to {start + truth_nz}")
        # print(f"extracting plane {need_plane}")
        for angle in range(len(angles)):
            for phase in range(nphases):
                print(f"plane: {plane}, angle: {angle}, phase: {phase}")
                # check_mem()
                temp = gpuarray.to_gpu(illum[angle, phase, start : start + truth_nz])
                # print("temp illum created ")
                # check_mem()
                temp = (temp * truth_gpu).astype(np.complex64)
                # print("multiplied by ground truth and converted to np64")
                # check_mem()
                do_fft(temp, temp, inverse=0)
                # print("temp fft performed ")
                # check_mem()
                temp = temp * otf_gpu
                # print("multiplied by otf")
                # check_mem()
                do_fft(temp, temp, inverse=1)
                # print("inverse fft ")
                # check_mem()
                do_fft_shift(temp, temp)
                temp = temp[need_plane].real.astype(np.float32)
                out[angle, phase, plane] = zoom(
                    temp, (1 / upscale_xy, 1 / upscale_xy), mode="cubic"
                ).get()
                # print("end of loop")
                # check_mem()
                del temp

    return out

    # cupy version
    # mempool = cp.get_default_memory_pool()
    # mempool.free_all_blocks()
    # out = np.empty((len(angles), nphases, out_nz, out_ny, out_nx), np.float32)
    # truth_gpu = cp.asarray(truth)
    # otf_gpu = cp.fft.fftn(cp.asarray(psf))
    # for plane in range(out_nz):
    #     start = plane * upscale_z
    #     need_plane = truth_nz - 1 - (upscale_z // 2 + (plane * upscale_z))
    #     print(f"illum_seg: {start} to {start + truth_nz}")
    #     print(f"extracting plane {need_plane}")
    #     for angle in range(len(angles)):
    #         for phase in range(nphases):
    #             print(f"plane: {plane}, angle: {angle}, phase: {phase}")
    #             res_gpu = cp.fft.ifftn(
    #                 cp.fft.fftn(
    #                     cp.multiply(
    #                         illum[angle, phase, start : start + truth_nz], truth_gpu
    #                     )
    #                 )
    #                 * otf_gpu
    #             ).real
    #             res_gpu = cp.fft.fftshift(res_gpu)[need_plane].astype(np.float32)
    #             out[angle, phase, plane] = zoom(
    #                 res_gpu, (1 / upscale_xy, 1 / upscale_xy)
    #             ).get()
    #             del res_gpu
    #             mempool.free_all_blocks()


if __name__ == "__main__":
    import tifffile as tf
    import matplotlib.pyplot as plt
    import mrc

    plt.ion()

    out = main()

    _out = np.transpose(out, (0, 2, 1, 4, 3)).reshape((-1, out.shape[3], out.shape[4]))
    _out = np.ascontiguousarray(np.fliplr(_out))
    # tf.imshow(_out)
    # plt.show(block=True)
    mrc.save(
        _out,
        "/Users/talley/Desktop/ss/py.dv",
        metadata={"wave0": 528, "dxy": 0.08, "dz": 0.125, "LensNum": 10612},
    )
    # mrc.save(
    #     _psf.astype("single"),
    #     "/Users/talley/Desktop/ss/psf_py.dv",
    #     metadata={"wave0": 528, "dxy": truth_dx, "dz": truth_dz, "LensNum": 10612},
    # )


# def compare():

#     im_mat = mrc.imread("/Users/talley/Desktop/ss/mat.dv")[:55][::5]
#     im_py = mrc.imread("/Users/talley/Desktop/ss/py.dv")[:55][::5]
#     im_matf = np.fft.fftshift(np.abs(np.fft.fftn(im_mat)))
#     im_pyf = np.fft.fftshift(np.abs(np.fft.fftn(im_py)))
#     tf.imshow(np.log2(im_pyf))
#     plt.show()
