import sage
from sage.all import matrix as sage_matrix

import numpy as np

from . import _numpy_wrappers as nwrap

def _vectorize(func):
    def vfunc(arr):
        flat_res = np.array([func(elt) for elt in arr.flatten()],
                            dtype=arr.dtype)
        return flat_res.reshape(arr.shape)
    return vfunc

def inexact_type(dtype):
    return not np.can_cast(dtype, int) and np.can_cast(dtype, float)

def apply_matrix_func(A, numpy_func, sage_func, expected_shape=None):
    if inexact_type(A.dtype):
        return numpy_func(A)

    return sage_matrix_func(A, sage_func, expected_shape)

def sage_matrix_func(A, sage_func, expected_shape=None):
    if expected_shape is None:
            expected_shape = A.shape

    mat_flat = list(A.reshape((-1,) + A.shape[-2:]))
    mat_res = [sage_func(sage_matrix(mat))
               for mat in mat_flat]

    return np.array(mat_res, dtype='O').reshape(expected_shape)

def sage_matrix_list(A):
    mat_flat = A.reshape((-1,) + A.shape[-2:])
    return [sage_matrix(mat) for mat in mat_flat]

def change_base_ring(arr, base_ring, rational_approx=False):
    arr = arr.astype('object')

    ring_convert = base_ring
    if rational_approx:
        ring_convert = (lambda x : base_ring(sage.all.QQ(x)))

    return _vectorize(ring_convert)(arr)

def invert(A):
    return apply_matrix_func(
        A, np.linalg.inv, lambda M : M.inverse()
    )

def kernel(A):
    return apply_matrix_func(
        A, nwrap.kernel, lambda M: M.right_kernel_matrix().T,
        expected_shape=(A.shape[:-2] + (A.shape[-1], A.shape[-1] - A.shape[-2]))
    )
