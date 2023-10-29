import inspect

from . import core as lie

def _has_inv_param(hom):
    params = inspect.signature(hom).parameters.values()
    for param in params:
        if (param.name == "inv" or
            param.kind == param.VAR_KEYWORD):
            return True
    return False

def _wrap_hom(hom, *args, **kwargs):
    if _has_inv_param(hom):
        def wrapped_hom(mat, inv=None):
            return hom(mat, *args, inv=inv, **kwargs)
        return wrapped_hom

    def wrapped_hom(mat, inv=None):
        return hom(mat, *args)
    return wrapped_hom

def sl2_irrep(n, **kwargs):
    return _wrap_hom(lie.sl2_irrep, n, **kwargs)

def sln_adjoint(**kwargs):
    return _wrap_hom(lie.sln_adjoint, **kwargs)

def gln_adjoint(**kwargs):
    return _wrap_hom(lie.gln_adjoint, **kwargs)

def sl2_to_so21():
    return _wrap_hom(lie.sl2_to_so21)
