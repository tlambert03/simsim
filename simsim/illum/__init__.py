# from simsim.illum_pycuda import structillum_3d_tex
# from pycuda import gpuarray
# import numpy as np


# class SIMIllum:
#     def __init__(
#         self,
#         shape,
#         dx,
#         dz,
#         angles,
#         nphases=5,
#         linespacing=0.2305,
#         pattern_defocus=0,
#         modulation_contrast=1,
#         NA=1.42,
#         nimm=1.515,
#         wvl=0.488,
#         model=structillum_3d_tex,
#     ):
#         self.genfunc = model
#         self.shape = shape
#         self.dx = dx
#         self.dz = dz
#         self.angles = angles
#         self.nphases = nphases
#         self.linespacing = linespacing
#         self.pattern_defocus = pattern_defocus
#         assert 0 <= modulation_contrast <= 1, "modulation_contrast must be from 0-1"
#         self.modulation_contrast = modulation_contrast
#         self.NA = NA
#         self.nimm = nimm
#         self.wvl = wvl
#         self._data = None

#     @property
#     def data(self):
#         if self._data is None:
#             return self.generate()
#         return self._data

#     def generate(self):
#         illum = self.genfunc(
#             self.shape,
#             self.angles,
#             self.nphases,
#             linespacing=self.linespacing,
#             dx=self.dx,
#             dz=self.dz,
#             defocus=self.pattern_defocus,
#             NA=self.NA,
#             nimm=self.nimm,
#             wvl=self.wvl,
#         )
#         if isinstance(illum, gpuarray.GPUArray):
#             illum = np.ascontiguousarray(illum.get())

#         _min = illum.min()
#         _max = illum.max()
#         # normalize
#         if not _min == 0 and _max == 1:
#             illum -= _min
#             illum *= 1 / _max
#         # adjust modulation contrast
#         if self.modulation_contrast < 1:
#             illum *= self.modulation_contrast
#             illum += 1 - self.modulation_contrast

#         self._data = illum
#         return illum
