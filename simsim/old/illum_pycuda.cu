#include <stdio.h>

texture<float, 2, cudaReadModeElementType> texref2d;

__global__ void illum3d(float *output, int nx, int ny, int nz, float texwidth, float *mat) {
  // Calculate oputput coordinates
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

  // Read from texture and write to global memory
  int idx = z * (nx * ny) + y * nx + x;
  
  // since address mode WRAP doesn't seem to be working...
  if (tu <= 0 || tu >= texwidth) {
    tu = abs(fmod(tu, texwidth));
  } 
  output[idx] = tex2D(texref2d, tu, tv);
}
