#include <cubicTex2D.cu>
#include <cubicTex3D.cu>
#include <stdio.h>

extern "C" {
texture<float, cudaTextureType2D, cudaReadModeElementType> texref2d;
texture<float, cudaTextureType3D, cudaReadModeElementType> texref3d;

__global__ void affine2D(float *output, int nx, int ny, float *mat,
                         bool cubic) {

  // Calculate texture coordinates
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= nx || y >= ny) {
    return;
  }

  float u = x;
  float v = y;

  float tu = mat[0] * u + mat[1] * v + mat[2] + 0.5f;
  float tv = mat[3] * u + mat[4] * v + mat[5] + 0.5f;

  // Read from texture and write to global memory
  int idx = y * nx + x;
  if (cubic) {
    output[idx] = cubicTex2D(texref2d, tu, tv);
  } else {
    output[idx] = tex2D(texref2d, tu, tv);
  }
}

__global__ void affine3D(float *output, int nx, int ny, int nz, float *mat,
                         bool cubic) {

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
  if (cubic) {
    output[idx] = cubicTex3D(texref3d, tu, tv, tw);
  } else {
    output[idx] = tex3D(texref3d, tu, tv, tw);
  }

  // if (x - 1 < nx / 2 && x + 1 > nx / 2 && y - 1 < ny / 2 && y + 1 > ny / 2) {
  //   printf("x: %f \n", output[idx]);
  // }
}

__global__ void affine2D_RA(float *output, int nx, int ny, float dx, float dy,
                            float *mat, bool cubic) {

  // Calculate texture coordinates
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= nx || y >= ny) {
    return;
  }

  float u = x;
  float v = y;

  // intrinsic coords to world
  u = 0.5 + (u - 0.5) * dx;
  v = 0.5 + (v - 0.5) * dy;

  // transform coordinates in world coordinate frame
  float tu = mat[0] * u + mat[1] * v + mat[2];
  float tv = mat[3] * u + mat[4] * v + mat[5];

  // world coords to intrinsic
  tu = 0.5 + (tu - 0.5) / dx;
  tv = 0.5 + (tv - 0.5) / dy;

  // Read from texture and write to global memory
  int idx = y * nx + x;
  if (cubic) {
    output[idx] = cubicTex2D(texref2d, tu, tv);
  } else {
    output[idx] = tex2D(texref2d, tu, tv);
  }
}

__global__ void affine3D_RA(float *output, int nx, int ny, int nz, float dx,
                            float dy, float dz, float *mat, bool cubic) {

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
  if (cubic) {
    output[idx] = cubicTex3D(texref3d, tu, tv, tw);
  } else {
    output[idx] = tex3D(texref3d, tu, tv, tw);
  }
}
}