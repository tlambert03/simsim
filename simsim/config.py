import configparser
import logging

import numpy as np

from .dosim import dosim
from .illum import SIMIllum
from .psf import PSF
from .truth import Truth

logger = logging.getLogger(__name__)


class Objective:
    def __init__(
        self,
        na=1.42,
        ni0=1.515,
        ni=1.515,
        ng0=1.515,
        ng=1.515,
        ns=1.515,
        tg0=170,
        tg=170,
        ti0=150,
    ):
        self.na = float(na)
        self.ni0 = float(ni0)
        self.ni = float(ni)
        self.ng0 = float(ng0)
        self.ng = float(ng)
        self.ns = float(ns)
        self.tg0 = float(tg0)
        self.tg = float(tg)
        self.ti0 = float(ti0)


class Config:
    """Converts an ini file into a collection of objects ready for simulation
    
    no computation should be performed
    """

    def __init__(self, config_file="simsim/default.ini"):
        self.parser = configparser.ConfigParser(inline_comment_prefixes="#")
        self.parser.read(config_file)
        self._truth = None
        self._illum = None
        self._channels = None
        self._objective = None
        self._psf = None
        self._camera = None

    @property
    def channels(self):
        if not self._channels:
            # TODO: enable multiple channels
            self._channels = [
                {
                    "exwave": self.parser.getfloat(
                        "CHANNELS", "exwave", fallback=0.488
                    ),
                    "emwave": self.parser.getfloat(
                        "CHANNELS", "emwave", fallback=0.525
                    ),
                }
            ]
            for c, chan in enumerate(self._channels):
                for k, v in chan.items():
                    if v < 0.200 or v > 1.5:
                        msg = (
                            "Invalid channel {} {}: {}um.  (should be 0.2 < wvl < 1.5)"
                        )
                        raise ValueError(msg.format(c, k, v))
        return self._channels

    @property
    def objective(self):
        if not self._objective:
            objective_dict = dict(
                (k, v)
                for k, v, in self.parser["OBJECTIVE"].items()
                if k not in self.parser.defaults()
            )
            self._objective = Objective(**objective_dict)
        return self._objective

    @property
    def psf(self):
        if self._psf is None:
            pz = self.parser.getfloat("PSF", "pz", fallback=0)
            model = self.parser.get("PSF", "model")
            self._psf = PSF(
                self.illum.shape,
                vars(self.objective),
                self.truth.dx,
                self.truth.dz,
                pz=pz,
                wvl=self.channels[0]["exwave"],
                model=model,
            )
        return self._psf

    @property
    def truth(self):
        if not self._truth:
            if "GROUND TRUTH" not in self.parser:
                raise configparser.MissingSectionHeaderError(
                    "No 'GROUND TRUTH' section"
                )
            section = self.parser["GROUND TRUTH"]
            source = section.get(
                "source", fallback="simsim.truth.matslines.matslines3D"
            )
            self.truth_nx = section.getint("nx")
            nz = section.getint("nz")
            nx = section.getint("nx")
            _standard = ["source", "nx", "nz", "dx", "dz", "scale_x", "scale_z"]
            _standard += self.parser.defaults().keys()
            extra_kwargs = {
                k: v for k, v in section.items() if k.lower() not in _standard
            }
            self._truth = Truth(
                source,
                shape=(nz, nx, nx),
                dx=section.getfloat("dx", 0.01),
                dz=section.getfloat("dz", 0.025),
                scale_x=section.getint("scale_x"),
                scale_z=section.getint("scale_z"),
                kwargs=extra_kwargs,
            )
        return self._truth

    @property
    def illum(self):
        if not self._illum:
            if "SIM" not in self.parser:
                return None
            section = self.parser["SIM"]
            angles = [float(x) for x in section.get("angles").split(",")]

            # we broaden the illumination in z to account for having to translate
            # the truth matrix through z during the z stack
            illum_shape = (
                self.truth.nz + self.truth.out_nz // 2 * self.truth.scale_z * 2,
                self.truth.nx,
                self.truth.nx,
            )

            self._illum = SIMIllum(
                illum_shape,
                self.truth.dx,
                self.truth.dz,
                angles,
                nphases=section.getint("nphases", 5),
                linespacing=section.getfloat("linespacing", 0.2035),
                pattern_defocus=section.getfloat("pattern_defocus", 0),
                modulation_contrast=section.getfloat("modulation_contrast", 1),
                NA=self.objective.na,
                nimm=self.objective.ni,
                wvl=self.channels[0]["exwave"],
            )
        return self._illum


class Simulation:
    def __init__(self, config):
        self.config = config
        self._result = None

    def run(self):
        truth = self.config.truth.data
        illum = self.config.illum.data
        psf = self.config.psf.data
        out = dosim(
            truth, illum, psf, self.config.truth.out_nz, self.config.truth.out_nx
        )
        out = np.transpose(out, (0, 2, 1, 4, 3)).reshape(
            (-1, out.shape[3], out.shape[4])
        )
        self._result = np.ascontiguousarray(np.fliplr(out))
        return self._result

    @property
    def result(self):
        if self._result is None:
            return self.run()
        return self._result
