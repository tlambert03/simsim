import numpy as np


def _mats_vertices(shape=(64, 256, 256), density=10, length=10, horiZ=10):
    nz, ny, nx = shape
    numlines = nx * int(density)

    alpha = np.random.rand(numlines) * 2 * np.pi  # random set of angles
    alphaz = (
        np.pi / 2 + np.random.rand(numlines) * np.pi / int(horiZ)
    )  # random set of angles

    xypad = 3
    zpad = 5
    # random set of x, y, z centers
    x1 = np.round(xypad + (nx - xypad) * np.random.rand(numlines))
    y1 = np.round(xypad + (ny - xypad) * np.random.rand(numlines))
    z1 = np.round(zpad + (nz - zpad) * np.random.rand(numlines))

    # find other end of line given alpha and length
    lens = nx / 20 + int(length) * ny / 20 * np.random.rand(numlines)
    x2 = np.maximum(
        np.minimum(np.round(x1 + np.sin(alphaz) * np.cos(alpha) * lens), nx), 2
    )
    y2 = np.maximum(
        np.minimum(np.round(y1 + np.sin(alphaz) * np.sin(alpha) * lens), ny), 2
    )
    z2 = np.maximum(np.minimum(np.round(z1 + np.cos(alphaz) * lens), nz), 2)
    return np.ascontiguousarray(np.stack([x1, y1, z1, x2, y2, z2]).T.astype(np.int32))


def _enforce_ellipse(array):
    nz, ny, nx = array.shape
    zcrd, ycrd, xcrd = np.mgrid[0:nz, 0:ny, 0:nx]
    outside = (
        np.sqrt(
            ((xcrd - nx / 2) ** 2 + (ycrd - ny / 2) ** 2) / ((nx / 2) ** 2)
            + 1.5 * (zcrd - nz / 2) ** 2 / ((nz / 2) ** 2)
        )
        >= 0.9
    )
    array[outside] = 0
    return array


def matslines3D(shape=(64, 256, 256), density=10, length=10, horiZ=10):
    from .bresenham import bresenham_3D
    
    vertices = _mats_vertices(shape, density, length, horiZ)
    max_depth = np.sqrt(np.square(shape).sum()) * 0.8
    coords = bresenham_3D(vertices, max_out=max_depth).reshape((-1, 3))
    unique, counts = np.unique(coords, axis=0, return_counts=True)
    # the first item will be the empty stuff returned from bresenham_3d
    assert np.all(unique[0] == -1)
    out = np.zeros(shape, "uint8")
    for (x, y, z), count in zip(unique[1:], counts[1:]):
        try:
            out[(z, y, x)] = count
        except IndexError:
            pass
    return _enforce_ellipse(out)


def bresenham_3D_cpu(p1, p2):
    """find coordinates on a 3-D line using Bresenham's Algorithm"""

    x1, y1, z1 = p1
    x2, y2, z2 = p2

    points_list = []
    points_list.append((int(x1), int(y1), int(z1)))
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    xs = 1 if x2 > x1 else -1
    ys = 1 if y2 > y1 else -1
    zs = 1 if z2 > z1 else -1

    # Driving axis is X-axis"
    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x1 != x2:
            x1 += xs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dx
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            points_list.append((int(x1), int(y1), int(z1)))

    # Driving axis is Y-axis"
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y1 != y2:
            y1 += ys
            if p1 >= 0:
                x1 += xs
                p1 -= 2 * dy
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            points_list.append((int(x1), int(y1), int(z1)))

    # Driving axis is Z-axis"
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z1 != z2:
            z1 += zs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dz
            if p2 >= 0:
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            points_list.append((int(x1), int(y1), int(z1)))
    return points_list


def matslines3D_cpu(shape=(64, 256, 256), density=10, length=10, horiZ=10):
    """generate 3D array of line segments in random orientations

    Args:
        shape (list, optional): shape of output array (nz, ny, nx)
        density (int, optional): [description]. Defaults to 10.
        length (int, optional): [description]. Defaults to 10.
        horiZ (int, optional): controls how "horizontal" the lines are in Z. Defaults to 10.
    """
    nz, ny, nx = shape
    vertices = _mats_vertices(shape, density, length, horiZ)
    out = np.zeros(shape, "uint8")
    for p1, p2 in vertices.reshape((-1, 2, 3)):
        for x, y, z in bresenham_3D_cpu(p1, p2):
            try:
                out[z, y, x] += 1
            except IndexError:
                pass

    zcrd, ycrd, xcrd = np.mgrid[0:nz, 0:ny, 0:nx]
    outside = (
        np.sqrt(
            ((xcrd - nx / 2) ** 2 + (ycrd - ny / 2) ** 2) / ((nx / 2) ** 2)
            + 1.5 * (zcrd - nz / 2) ** 2 / ((nz / 2) ** 2)
        )
        >= 0.9
    )
    out[outside] = 0
    return out
