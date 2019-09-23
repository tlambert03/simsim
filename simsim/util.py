def get_callable(string):
    from importlib import import_module

    p, m = string.rsplit(".", 1)
    mod = import_module(p)
    met = getattr(mod, m)
    if callable(met):
        return met
    raise ImportError("Could not import string '{}' as a callable".format(string))