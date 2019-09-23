import numpy as np
from pycuda import gpuarray
# import simsim.cuda.initpycuda
from reikna import fft
from reikna.cluda.cuda import Thread
from skimage.transform import downscale_local_mean as downscale
from tqdm import tqdm


def dosim(truth, illum, psf, out_nz, out_nx):
    truth_nz = truth.shape[0]
    truth_nx = truth.shape[-1]
    upscale_z = truth_nz // out_nz
    upscale_xy = truth_nx // out_nx

    nangles = illum.shape[0]
    nphases = illum.shape[1]
    truth_gpu = gpuarray.to_gpu(truth)
    otf_gpu = gpuarray.to_gpu(psf.astype(np.complex64))

    thr = Thread(simsim.cuda.initpycuda.context)
    do_fft = fft.FFT(otf_gpu[0]).compile(thr, fast_math=True)
    do_fft_shift = fft.FFTShift(otf_gpu[0]).compile(thr, fast_math=True)

    for o in otf_gpu:
        do_fft(o, o, inverse=0)

    def conv2(truthplane, otf_plane):
        # otf_plane is wrong...
        do_fft(truthplane, truthplane, inverse=0)
        truthplane *= otf_gpu[otf_plane]
        do_fft(truthplane, truthplane, inverse=1)
        do_fft_shift(truthplane, truthplane)
        return truthplane.real

    out = np.empty((nangles, nphases, out_nz, out_nx, out_nx), np.float32)
    with tqdm(total=(nangles * nphases * out_nz)) as pbar:
        for z in range(out_nz):
            start = z * upscale_z
            psfplane = truth_nz - 1 - (upscale_z // 2 + start) + psf.shape[0] // 2
            for a in range(nangles):
                for p in range(nphases):

                    emissionmap = gpuarray.to_gpu(illum[a, p, start : start + truth_nz])
                    emissionmap *= truth_gpu
                    _sum = gpuarray.zeros((truth_nx, truth_nx), np.float32)

                    for P, truthplane in enumerate(emissionmap):
                        if gpuarray.sum(truthplane).get() > 0:
                            C = conv2(truthplane.astype(np.complex64), P - psfplane)
                            _sum += C
                    out[a, p, z] = downscale(_sum.get(), (upscale_xy, upscale_xy))
                    pbar.update(1)
    return out


# if a == 0 and p == 0:
#     plt.imshow(emissionmap[].get())
#     fig.canvas.draw()
#     fig.canvas.flush_events()

# import simsim.cuda.initpycuda
from pycuda import gpuarray
from reikna import fft
from reikna.cluda.cuda import Thread
import numpy as np
from tqdm import tqdm


def convolve(truth, psf):
    thr = Thread(simsim.cuda.initpycuda.context)
    truth_gpu = gpuarray.to_gpu(truth.astype(np.complex64))
    otf_gpu = gpuarray.to_gpu(psf.astype(np.complex64))
    do_fft = fft.FFT(otf_gpu).compile(thr, fast_math=True)
    do_fft_shift = fft.FFTShift(otf_gpu).compile(thr, fast_math=True)
    do_fft(otf_gpu, otf_gpu, inverse=0)
    do_fft(truth_gpu, truth_gpu, inverse=0)
    truth_gpu *= otf_gpu
    do_fft(truth_gpu, truth_gpu, inverse=1)
    do_fft_shift(truth_gpu, truth_gpu)
    return truth_gpu.get()


def dosim_gpu(illum, truth, psf, upscale_z, upscale_x):
    nangles = illum.shape[0]
    nphases = illum.shape[1]
    truth_nz = truth.shape[0]
    out_nz = truth_nz // upscale_z
    out_ny = truth.shape[1] // upscale_x
    out_nx = truth.shape[2] // upscale_x
    out = np.empty((nangles, nphases, out_nz, out_ny, out_nx), np.float32)

    thr = Thread(simsim.cuda.initpycuda.context)
    truth_gpu = gpuarray.to_gpu(truth)
    otf_gpu = gpuarray.to_gpu(psf.astype(np.complex64))
    do_fft = fft.FFT(otf_gpu).compile(thr, fast_math=True)
    do_fft_shift = fft.FFTShift(otf_gpu).compile(thr, fast_math=True)
    do_fft(otf_gpu, otf_gpu, inverse=0)

    with tqdm(total=(out_nz * nangles * nphases)) as pbar:
        for z in range(out_nz):
            start = z * upscale_z
            need_plane = truth_nz - 1 - (upscale_z // 2 + (z * upscale_z))
            print(
                f"illum_seg: {start} to {start + truth_nz}... extracting plane {need_plane}"
            )
            for a in range(nangles):
                for p in range(nphases):
                    temp = gpuarray.to_gpu(illum[a, p, start : start + truth_nz])
                    temp = (temp * truth_gpu).astype(np.complex64)
                    do_fft(temp, temp, inverse=0)
                    temp = temp * otf_gpu
                    do_fft(temp, temp, inverse=1)
                    do_fft_shift(temp, temp)
                    temp = temp[need_plane].real.astype(np.float32)
                    out[a, p, z] = downscale(temp.get(), (upscale_x, upscale_x))
                    # out[a, p, z] = zoom(temp, (1 / upscale_x, 1 / upscale_x), mode="cubic").get()
                    pbar.update(1)
                    del temp
    return out
    