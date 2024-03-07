import warnings

import numpy as np

from .. import representation
from . import core

if not core.SAGE_AVAILABLE:
    raise ImportError("geometry_tools.utils.snappy cannot be imported outside of a working Sage environment")

from . import sagewrap

def exact_holonomy(manifold, sage_matrices=True, prec=100,
                   degree=10, optimize=True, **kwargs):
    """Get the holonomy representation of a hyperbolic 3-manifold, as a
    representation into SL(2, F) where F is an algebraic number field.

    This function uses Snappy to obtain a numerical approximation of
    the holonomy representation for the given 3-manifold. Then it uses
    Snappy's built-in Sage integration to compute a guess for an
    algebraic number field L containing the holonomy matrix
    entries. Each generator for the holonomy is converted to a Sage
    matrix with entries lying in L.

    The resulting representation is returned as a
    geometry_tools.representation.SageMatrixRepresentation object (or
    simply as a geometry_tools.representation.Representation, if
    desired).

    For practical purposes, it may be simpler to change the base ring
    of this representation to QQbar (Sage's implementation of the
    algebraic closure of the rationals), especially if you plan to
    compose the SL(2, C) representation with other Lie group
    homomorphisms (e.g. the isomorphism to SO(3,1) or the adjoint
    representation).


    Parameters
    ----------
    manifold : Manifold
        A Snappy manifold object.
    sage_matrices : bool
        If True (default), return a SageMatrixRepresentation instead
        of a plain Representation object. (Note that
        SageMatrixRepresentations can be freely cast to
        Representations without losing data, and vice-versa.)

    I would refer to documentation for the Sage/Snappy find_field
    function for the "prec", "degree", and "optimize" parameters, but
    this documentation apparently doesn't exist.

    Returns
    -------
    SageMatrixRepresentation or Representation
        Holonomy representation for the given manifold, as computed by
        Snappy

    """
    entries = manifold.holonomy_matrix_entries()
    gp = manifold.fundamental_group()
    K, _, polys = entries.find_field(prec=prec, degree=degree,
                                     optimize=optimize, **kwargs)
    poly_mats = np.array(polys).reshape((-1, 2, 2))

    rep = representation.SageMatrixRepresentation(relations=gp.relators(),
                                                  base_ring=K)

    for g, mat in zip(gp.generators(), poly_mats):
        rep[g] = sagewrap.sage_matrix(K, mat)

    return rep
