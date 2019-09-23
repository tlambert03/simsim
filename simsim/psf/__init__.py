from psfmodels import vectorial_psf_centered
import numpy as np
import logging
from ..util import get_callable

logger = logging.Logger(__name__)


class PSF:
    def __init__(self, shape, params, dx=0.01, dz=0.025, pz=0, wvl=0.488, model=None):
        self.shape = shape
        self.nz, self.ny, self.nx = self.shape
        self.params = params
        self.dx = dx
        self.dz = dz
        self.pz = pz
        self.wvl = wvl
        self.model = vectorial_psf_centered
        if model is not None:
            if callable(model):
                self.model = model
            elif isinstance(model, str):
                self.model = get_callable(model)
            else:
                raise ValueError(
                    'PSF model "{}" not a callable or string'.format(model)
                )
        self._data = None

    @property
    def data(self):
        if self._data is None:
            return self.generate()
        return self._data

    def generate(self):
        trimz = -(self.nz % 2 - 1)
        trimx = -(self.nx % 2 - 1)
        psf_nz = self.nz - trimz
        psf_nx = self.nx - trimx
        if 'na' in self.params:
            self.params["NA"] = self.params.pop("na")
        psf = self.model(
            psf_nz,
            dz=self.dz,
            nx=psf_nx,
            dxy=self.dx,
            pz=self.pz,
            wvl=self.wvl,
            params=self.params,
        )
        psf /= psf.sum()
        psf = np.pad(psf, ((trimz, 0), (trimx, 0), (trimx, 0)))
        self._data = psf
        return psf
