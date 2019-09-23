import os
import numpy as np


class Truth:
    """[summary]

    if source is a callable, it should able to convert all values in `kwargs`
    from strings to the desired type for the callable.

    Raises:
        NotImplementedError: [description]
        ValueError: [description]
        ValueError: [description]
        ValueError: [description]
        ValueError: [description]

    Returns:
        [type]: [description]
    """

    def __init__(
        self,
        source,
        shape=(65, 1024, 1024),
        dx=0.01,
        dz=0.025,
        scale_x=8,
        scale_z=5,
        kwargs=None,
    ):
        self._data = None
        self.genfunc = None
        self.shape = shape
        if isinstance(source, np.ndarray):
            self._data = source
        elif callable(source):
            self.genfunc = source
        elif isinstance(source, str):
            _p = os.path.abspath(os.path.expanduser(source))
            if os.path.exists(_p):
                if _p.lower().endswith((".tif", ".tiff")):
                    import tifffile as tf

                    self._data = tf.imread(_p)
                elif _p.lower().endswith((".mrc", ".dv")):
                    import mrc

                    self._data = mrc.imread(_p)
                else:
                    ext = os.path.splitext(_p)[-1]
                    raise NotImplementedError(
                        "Truth source argument was recognized as a file path, but no "
                        "reader is available for extension type: '{}'".format(ext)
                    )
            else:
                from importlib import import_module

                try:
                    p, m = source.rsplit(".", 1)
                    mod = import_module(p)
                    met = getattr(mod, m)
                    if callable(met):
                        self.genfunc = met
                    else:
                        raise Exception()
                except Exception:
                    raise ValueError(
                        "Value provided for Truth source ('{}') ".format(source)
                        + "could not be interpreted either as a filepath "
                        + "or an importable callable"
                    )
        else:
            raise ValueError("Unrecognized value for Truth source")
        if self._data is not None:
            self.shape = self._data.shape
        self.dx = dx
        self.dz = dz
        self.nz, self.ny, self.nx = self.shape

        _s = "Truth n{d} ({nd}) must be an even multiple of scale_{d} ({sd})"
        if self.nx % scale_x:
            raise ValueError(_s.format(**dict(d="x", nd=self.nx, sd=scale_x)))
        if self.nz % scale_z:
            raise ValueError(_s.format(**dict(d="z", nd=self.nz, sd=scale_z)))
        self.scale_x = scale_x
        self.scale_z = scale_z
        self.out_nx = self.nx // scale_x
        self.out_nz = self.nz // scale_z

        if kwargs is not None:
            assert isinstance(kwargs, dict), "'kwargs' argument must be a dict"
            self.kwargs = kwargs
        else:
            self.kwargs = dict()

    @property
    def data(self):
        if self._data is None:
            return self.generate()
        return self._data

    def generate(self):
        if not callable(self.generate):
            raise ValueError(
                "Cannot generate. Truth was not instantiated with a callable function"
            )
        self._data = self.genfunc(self.shape, **self.kwargs)
        return self._data
