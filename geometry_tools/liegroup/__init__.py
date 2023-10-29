import numpy as np

from ..base import GeometryError
from .. import utils

if utils.SAGE_AVAILABLE:
    from ..utils import sagewrap

def _convert_vector(vector, like=None, autoconvert=True):
    is_sage_obj = False
    if like is not None and utils.SAGE_AVAILABLE:
        is_sage_obj = (isinstance(like, sagewrap.matrix_class) or
                        isinstance(like, sagewrap.vector_class))
    if (autoconvert and
        utils.SAGE_AVAILABLE and
        is_sage_obj):

        if len(vector.shape) == 1:
            return sagewrap.sage_vector(vector)

        return sagewrap.sage_vector_list(vector)

    return vector

def _convert_matrix(matrix, like=None, autoconvert=True):
    is_sage_obj = False
    if like is not None and utils.SAGE_AVAILABLE:
        is_sage_obj = (isinstance(like, sagewrap.matrix_class) or
                        isinstance(like, sagewrap.vector_class))
    if (autoconvert and
        utils.SAGE_AVAILABLE and
        is_sage_obj):
        if len(matrix.shape) == 2:
            return sagewrap.sage_matrix(matrix)
        return sagewrap.sage_matrix_list(matrix)

    return matrix

def basis_matrix(i, j, n, as_sage=False, **kwargs):
    bm = utils.zeros((n,n), **kwargs)
    one = utils.number(1, **kwargs)
    bm[i,j] = one

    if as_sage:
        return sagewrap.sage_matrix(bm)
    return bm

def sln_basis_matrix(i, j, n, as_sage=False, **kwargs):
    bm = basis_matrix(i, j, n, as_sage=as_sage, **kwargs)
    if i == j:
        one = utils.number(1, **kwargs)
        bm[n-1, n-1] = -one
    return bm


def gln_lie_algebra_coords(matrix, dtype=None,
                           autoconvert=True):

    arr = np.array(matrix, dtype=dtype)
    n = arr.shape[-1]

    coords = arr.reshape(arr.shape[:-2] + (n*n,))
    return _convert_vector(coords, like=matrix,
                           autoconvert=autoconvert)

def coords_to_gln_algebra(coord_vector, dtype=None,
                          autoconvert=True):
    arr = np.array(coord_vector, dtype=dtype)
    d = arr.shape[-1]

    n = np.sqrt(d)
    if d != int(d):
        raise GeometryError(
            "Cannot interpret a vector with non-square length as "
            "an element in a square matrix space"
        )

    mats = arr.reshape((-1, n, n))
    return _convert_matrix(mats, like=coord_vector,
                           autoconvert=autoconvert)


def sln_lie_algebra_coords(matrix, dtype=None,
                           autoconvert=True):
    """Compute an explicit isomorphism between the space of traceless
    square matrices with coefficients in a ring R and the module
    R^(n^2 - 1).

    Parameters
    ----------
    matrix : array-like
        A traceless (n x n) matrix. Must be interpretable as a numpy
        ndarray (so Sage matrices are allowed).
    dtype : dtype
        The underlying dtype of the returned vector. If None, use the
        dtype of matrix.
    autoconvert : bool
        If True, and the input matrix is a Sage matrix, then convert
        the output to a Sage vector.

    Returns
    -------
    vector : ndarray or Sage vector
        Array with shape (..., n^2 - 1).
    """
    arr = np.array(matrix, dtype=dtype)
    n = arr.shape[-1]

    coords = arr.reshape(arr.shape[:-2] + (n*n,))[..., :-1]

    return _convert_vector(coords, like=matrix,
                           autoconvert=autoconvert)

def coords_to_sln_lie_algebra(coord_vector, dtype=None,
                              autoconvert=True):

    arr = np.array(coord_vector, dtype=dtype)
    aug_arr = np.concatenate([arr, utils.zeros(arr.shape[:-1] + (1,),
                                               like=coord_vector,
                                               dtype=dtype)],
                             axis=-1)

    gln_coords = coords_to_gln_lie_algebra(aug_arr, dtype=dtype,
                                           autoconvert=False)

    gln_coords[..., -1, -1] = -np.trace(gln_coords, axis1=-1, axis2=-2)

    return _convert_matrix(gln_coords, like=coord_vector,
                           autoconvert=autoconvert)

def linear_matrix_action(linear_map, n,
                         dtype=None,
                         base_ring=None,
                         like=None,
                         wrap_func=None,
                         unwrap_func=None):

    base_ring, dtype = utils.check_type(base_ring, dtype, like)
    map_matrix = utils.zeros((n*n, n*n), base_ring, dtype)

    for i in range(n):
        for j in range(n):
            bm = basis_matrix(i, j, n, like=map_matrix)

            if wrap_func is not None:
                bm = wrap_func(bm)

            b_image = linear_map(bm)

            if unwrap_func is not None:
                b_image = unwrap_func(b_image)

            map_matrix[:, i*n + j] = gln_lie_algebra_coords(
                b_image, autoconvert=False
            )

    if wrap_func is not None:
        map_matrix = wrap_func(map_matrix)

    return map_matrix

def sln_linear_action(linear_map, n,
                      dtype=None, base_ring=None, like=None,
                      wrap_func=None, unwrap_func=None, **kwargs):

    base_ring, dtype = utils.check_type(base_ring, dtype, like)
    map_matrix = utils.zeros((n**2 - 1, n**2 - 1), base_ring, dtype)

    for i in range(n):
        for j in range(n):
            if i == n - 1 and j == n - 1:
                break

            bm = sln_basis_matrix(i, j, n, like=map_matrix)

            if wrap_func is not None:
                bm = wrap_func(bm)

            b_image = linear_map(bm)

            if unwrap_func is not None:
                b_image = unwrap_func(b_image)

            map_matrix[:, i*n + j] = sln_lie_algebra_coords(
                b_image, autoconvert=False
            )

    if wrap_func is not None:
        map_matrix = wrap_func(map_matrix)

    return map_matrix
