import numpy as np
from sage.all import matrix as sage_matrix

def apply_matrix_func(A, numpy_func, sage_func, expected_shape=None):
    try:
        return numpy_func(A)
    except np.core._exceptions.UFuncTypeError as e:
        if expected_shape is None:
            expected_shape = A.shape

        mat_flat = list(A.reshape((-1,) + A.shape[-2:]))
        mat_res = [sage_func(sage_matrix(mat))
                   for mat in mat_flat]

        return np.array(mat_res, dtype='O').reshape(expected_shape)

def invert(A):
    return apply_matrix_func(
        A, np.linalg.inv, lambda M : M.inverse()
    )
