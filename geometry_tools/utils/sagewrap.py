try:
    import sage
    SAGE_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_AVAILABLE = False

import numpy as np
from sage.all import matrix as sage_matrix

def apply_matrix_func(A, numpy_func, sage_func, expected_shape=None):
    try:
        return numpy_func(A)
    except (np.core._exceptions.UFuncTypeError, TypeError) as e:
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

def invert(A):
    return apply_matrix_func(
        A, np.linalg.inv, lambda M : M.inverse()
    )

def check_dtype(base_ring, dtype, default_dtype='float64'):
    if base_ring is not None:
        if not SAGE_AVAILABLE:
            raise EnvironmentError(
                "Cannot specify base_ring unless running within sage"
            )
        if dtype is not None and dtype != np.dtype('object'):
            raise TypeError(
                "Cannot specify base_ring and dtype unless dtype is dtype('object')"
            )
        dtype = np.dtype('object')
    elif dtype is None:
        dtype = default_dtype

    return (base_ring, dtype)

def change_base_ring(arr, base_ring, inplace=False):
    if not SAGE_AVAILABLE:
        raise EnvironmentError(
            "Cannot set base_ring unless running within sage"
        )

    arr = arr.astype('object', copy=(not inplace))
    if not inplace:
        return np.frompyfunc(base_ring, 1, 1)(arr)

    arr[...] = np.frompyfunc(base_ring, 1, 1)(arr)
    return arr
