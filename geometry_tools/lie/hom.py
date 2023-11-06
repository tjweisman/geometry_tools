import inspect

from . import core as lie
from .. import utils

def _has_inv_param(hom):
    params = inspect.signature(hom).parameters.values()
    for param in params:
        if param.name == "inv":
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

def sl2_to_so21(**kwargs):
    return _wrap_hom(lie.sl2_to_so21, **kwargs)

def so21_to_sl2(**kwargs):
    return _wrap_hom(lie.o_to_pgl, **kwargs)

def slc_to_slr(**kwargs):
    return _wrap_hom(lie.slc_to_slr, **kwargs)

def sl2c_to_so31(**kwargs):
    return _wrap_hom(lie.sl2c_to_so31, **kwargs)

def block_include(n, **kwargs):
    return _wrap_hom(lie.block_include, n, **kwargs)

def so_adjoint(p, q=0, **kwargs):
    form = utils.indefinite_form(p, q, **kwargs)
    return form_adjoint(form, **kwargs)

def sp_adjoint(n, **kwargs):
    form = utils.symplectic_form(n, **kwargs)
    return form_adjoint(form, **kwargs)

def so21_adjoint(**kwargs):
    return so_adjoint(2, 1, **kwargs)

def form_adjoint(form, **kwargs):
    # here we don't just wrap the corresponding function from lie.core
    # since we can precompute some stuff
    lie_alg_kernel_mat = lie.bilinear_form_differential(form, **kwargs)
    lie_alg_basis = utils.kernel(lie_alg_kernel_mat, **kwargs)

    def hom(mat, inv=None):
        gln_adjoint_mat = lie.gln_adjoint(mat, inv=inv, **kwargs)
        return lie.subspace_action(gln_adjoint_mat,
                                   lie_alg_basis,
                                   **kwargs)
    return hom
