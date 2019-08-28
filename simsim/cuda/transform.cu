//#include <stdio.h>

texture<float, cudaTextureType3D, cudaReadModeElementType> texRef;

// Simple transformation kernel
__global__ void transformKernel(float *output, int nx, int ny, int nz,
                                float *mat) {

  // Calculate texture coordinates
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= nx || y >= ny || z >= nz) {
    return;
  }

  float u = x;
  float v = y;
  float w = z;

  float tu = mat[0] * u + mat[1] * v + mat[2] * w + mat[3] + 0.5f;
  float tv = mat[4] * u + mat[5] * v + mat[6] * w + mat[7] + 0.5f;
  float tw = mat[8] * u + mat[9] * v + mat[10] * w + mat[11] + 0.5f;

  // Read from texture and write to global memory
  int idx = z * (nx * ny) + y * nx + x;
  output[idx] = tex3D(texRef, tu, tv, tw);

  // if (x - 1 < nx / 2 && x + 1 > nx / 2 && y - 1 < ny / 2 && y + 1 > ny / 2){
  //     printf("x: %d y: %d z: %d; tu: %f tv: %f tw: %f\\n", x, y, z, tu, tv,
  //     tw);
  // }
}

// Simple transformation kernel
__global__ void transformKernelRA(float *output, int nx, int ny, int nz,
                                  float dx, float dy, float dz, float *mat) {

  // Calculate texture coordinates
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= nx || y >= ny || z >= nz) {
    return;
  }

  float u = x;
  float v = y;
  float w = z;

  // intrinsic coords to world
  u = 0.5 + (u - 0.5) * dx;
  v = 0.5 + (v - 0.5) * dy;
  w = 0.5 + (w - 0.5) * dz;

  // transform coordinates in world coordinate frame
  float tu = mat[0] * u + mat[1] * v + mat[2] * w + mat[3];
  float tv = mat[4] * u + mat[5] * v + mat[6] * w + mat[7];
  float tw = mat[8] * u + mat[9] * v + mat[10] * w + mat[11];

  // world coords to intrinsic
  tu = 0.5 + (tu - 0.5) / dx;
  tv = 0.5 + (tv - 0.5) / dy;
  tw = 0.5 + (tw - 0.5) / dz;

  // Read from texture and write to global memory
  int idx = z * (nx * ny) + y * nx + x;
  output[idx] = tex3D(texRef, tu, tv, tw);
}