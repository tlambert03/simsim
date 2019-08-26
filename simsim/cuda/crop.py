from pycuda.compiler import SourceModule


mod_crop = SourceModule(
    """
    __global__ void crop_kernel(float *in, int nx, int ny, int nz, int new_nx,
                                int new_ny, int new_nz, float *out) {
        unsigned xout = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned yout = blockIdx.y;
        unsigned zout = blockIdx.z;

        if (xout < new_nx) {
            // Assumption: new dimensions are <= old ones
            unsigned xin = xout + nx - new_nx;
            unsigned yin = yout + ny - new_ny;
            unsigned zin = zout + nz - new_nz;
            unsigned indout = zout * new_nx * new_ny + yout * new_nx + xout;
            unsigned indin = zin * nx * ny + yin * nx + xin;
            out[indout] = in[indin];
        }
    }
    """
)

crop = mod_crop.get_function("crop_kernel")
