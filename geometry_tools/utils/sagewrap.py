from sage.all import matrix as sage_matrix

import numpy as np

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

def change_base_ring(arr, base_ring, inplace=False):
    arr = arr.astype('object', copy=(not inplace))
    if not inplace:
        return np.frompyfunc(base_ring, 1, 1)(arr)

    arr[...] = np.frompyfunc(base_ring, 1, 1)(arr)
    return arr
