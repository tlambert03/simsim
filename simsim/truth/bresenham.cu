__global__ void bresenham3D(int nrows, int depth, int *points, int *out) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= nrows)
    return;

  // setup indexing
  int x1 = points[(row * 6)];
  int y1 = points[(row * 6) + 1];
  int z1 = points[(row * 6) + 2];
  int x2 = points[(row * 6) + 3];
  int y2 = points[(row * 6) + 4];
  int z2 = points[(row * 6) + 5];
  int outN = 0;
  int width = 3;
  int offset = (row * width) + (outN * nrows * width);

  int dx = x2 - x1;
  int dy = y2 - y1;
  int dz = z2 - z1;

  //  absolute values
  dx = dx < 0 ? -dx : dx;
  dy = dy < 0 ? -dy : dy;
  dz = dz < 0 ? -dz : dz;

  int xs = x2 > x1 ? 1 : -1;
  int ys = y2 > y1 ? 1 : -1;
  int zs = z2 > z1 ? 1 : -1;

  // add the first point
  out[offset] = x1;
  out[offset + 1] = y1;
  out[offset + 2] = z1;
  // outN must be incremented every time a point is added
  outN += 1;
  offset = (row * width) + (outN * nrows * width);

  // Driving axis is X-axis
  if (dx >= dy && dx >= dz) {
    int p1 = 2 * dy - dx;
    int p2 = 2 * dz - dx;
    while (x1 != x2) {
      x1 += xs;
      if (p1 >= 0) {
        y1 += ys;
        p1 -= 2 * dx;
      }
      if (p2 >= 0) {
        z1 += zs;
        p2 -= 2 * dx;
      }
      p1 += 2 * dy;
      p2 += 2 * dz;
      // add point
      out[offset] = x1;
      out[offset + 1] = y1;
      out[offset + 2] = z1;
      outN += 1;
      if (outN >= depth)
        return;
      offset = (row * width) + (outN * nrows * width);
    }
  } else if (dy >= dx && dy >= dz) {
    // Driving axis is Y-axis"
    int p1 = 2 * dx - dy;
    int p2 = 2 * dz - dy;
    while (y1 != y2) {
      y1 += ys;
      if (p1 >= 0) {
        x1 += xs;
        p1 -= 2 * dy;
      }
      if (p2 >= 0) {
        z1 += zs;
        p2 -= 2 * dy;
      }
      p1 += 2 * dx;
      p2 += 2 * dz;
      // add point
      out[offset] = x1;
      out[offset + 1] = y1;
      out[offset + 2] = z1;
      outN += 1;
      if (outN >= depth)
        return;
      offset = (row * width) + (outN * nrows * width);
    }
  } else {

    int p1 = 2 * dy - dz;
    int p2 = 2 * dx - dz;
    while (z1 != z2) {
      z1 += zs;
      if (p1 >= 0) {
        y1 += ys;
        p1 -= 2 * dz;
      }
      if (p2 >= 0) {
        x1 += xs;
        p2 -= 2 * dz;
      }
      p1 += 2 * dy;
      p2 += 2 * dx;
      // add point
      out[offset] = x1;
      out[offset + 1] = y1;
      out[offset + 2] = z1;
      outN += 1;
      if (outN >= depth)
        return;
      offset = (row * width) + (outN * nrows * width);
    }
  }
}