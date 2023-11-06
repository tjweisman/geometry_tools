import warnings

import sage
from sage.all import matrix as sage_matrix
from sage.all import vector as sage_vector

import numpy as np

from . import numerical
from . import types

I = sage.all.I
pi = sage.all.pi
Integer = sage.all.Integer
SR = sage.all.SR
matrix_class = sage.matrix.matrix0.Matrix
vector_class = sage.structure.element.Vector

def ndarray_mat(mat, **kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            category=PendingDeprecationWarning
        )
        return np.array(mat, **kwargs)

def _vectorize(func):
    def vfunc(arr):
        flat_res = ndarray_mat([func(elt) for elt in arr.flatten()],
                               dtype=arr.dtype)
        return flat_res.reshape(arr.shape)
    return vfunc

def _sage_eig(mat):
    D, P = mat.eigenmatrix_right()
    return (D.diagonal(), P)

    #return not np.can_cast(dtype, int) and (
    #    np.can_cast(dtype, np.dtype("complex")) or
    #    np.can_cast(dtype, float)
    #)

def apply_matrix_func(A, numpy_func, sage_func, expected_shape=None):
    if is_linalg_type(A):
        return numpy_func(A)

    return sage_matrix_func(A, sage_func, expected_shape)

def sage_matrix_func(A, sage_func, expected_shape=None):
    if expected_shape is None:
            expected_shape = A.shape

    mat_flat = list(A.reshape((-1,) + A.shape[-2:]))
    mat_res = [sage_func(sage_matrix(mat))
               for mat in mat_flat]

    return ndarray_mat(mat_res, dtype='O').reshape(expected_shape)

def sage_vector_list(v):
    v_flat = v.reshape((-1, v.shape[-1]))
    return [sage_vector(vec) for vec in v_flat]

def sage_matrix_list(A):
    mat_flat = A.reshape((-1,) + A.shape[-2:])
    return [sage_matrix(mat) for mat in mat_flat]

def guess_base_ring(arr):
    return sage_vector(arr.flatten()).base_ring()

def change_base_ring(arr, base_ring, rational_approx=False):
    if base_ring is None:
        return arr

    arr = np.array(arr, dtype='object')

    ring_convert = base_ring
    if rational_approx:
        ring_convert = (lambda x : base_ring(sage.all.QQ(x)))

    if arr.shape == ():
        return ring_convert(arr.flatten()[0])

    return _vectorize(ring_convert)(arr)

def invert(A):
    return apply_matrix_func(
        A, np.linalg.inv, lambda M : M.inverse()
    )

def kernel(A, assume_full_rank=False, matching_rank=True,
           with_dimensions=False, with_loc=False, **kwargs):

    if assume_full_rank and not matching_rank:
        raise ValueError("matching_rank must be True if assume_full_rank is True")

    if is_linalg_type(A):
        return numerical.svd_kernel(A, assume_full_rank=assume_full_rank,
                                    matching_rank=matching_rank,
                                    with_dimensions=with_dimensions,
                                    with_loc=with_loc, **kwargs)

    A_arr = ndarray_mat(A, dtype="O")
    orig_shape = A_arr.shape[:-2]

    matrix_list = sage_matrix_list(A_arr)

    #for stupid sage/numpy compatibility reasons, DON'T transpose
    #these until later
    kernel_mats = [
        M.right_kernel_matrix()
        for M in matrix_list
    ]
    try:
        _ = ndarray_mat(kernel_mats)
        actual_ranks_match = True
    except ValueError:
        actual_ranks_match = False

    if matching_rank and not actual_ranks_match:
        raise ValueError(
                "Input matrices do not have matching rank. Try calling "
                "this function with matching_rank=False."
            )

    matrix_arr = ndarray_mat(kernel_mats, dtype="O")
    if matching_rank:
        matrix_arr = matrix_arr.swapaxes(-1,-2)
        return matrix_arr.reshape(orig_shape + matrix_arr.shape[-2:])

    kernel_dims = np.array([
        M.dimensions()[0] for M in kernel_mats
    ]).reshape(orig_shape)

    matrix_arr = matrix_arr.reshape(orig_shape)

    possible_dims = np.unique(kernel_dims)
    kernel_bases = []
    kernel_dim_loc = []

    for kernel_dim in possible_dims:
        where_dim = (kernel_dims == kernel_dim)
        kernel_bases.append(
            ndarray_mat(list(matrix_arr[where_dim]), dtype="O").swapaxes(-1, -2)
        )
        kernel_dim_loc.append(where_dim)

    # flat is better than nested, still
    if not with_loc and not with_dimensions:
        return tuple(kernel_bases)

    if with_dimensions and not with_loc:
        return (possible_dims, tuple(kernel_bases))

    if with_loc and not with_dimensions:
        return (tuple(kernel_bases), tuple(kernel_dim_loc))

    return (possible_dims, tuple(kernel_bases), tuple(kernel_dim_loc))

def eig(A):
    if is_linalg_type(A):
        return np.linalg.eig(A)

    mat_flat = list(A.reshape((-1,) + A.shape[-2:]))

    eig_res = [_sage_eig(sage_matrix(mat)) for mat in mat_flat]
    eigvals, eigvecs = zip(*eig_res)

    return (ndarray_mat(eigvals).reshape(A.shape[:-2] + (A.shape[-1],)),
            ndarray_mat(eigvecs).reshape(A.shape))

def eigh(A):
    if is_linalg_type(A):
        return np.linalg.eigh(A)

    return eig(A)

def det(A):
    return apply_matrix_func(
        A, np.linalg.det, lambda M: M.det(),
        expected_shape=A.shape[:-2]
    )

def real(arr):
    return _vectorize(sage.all.real)(arr)

def imag(arr):
    return _vectorize(sage.all.imag)(arr)
