import sage
from sage.all import matrix as sage_matrix
from sage.all import vector as sage_vector

import numpy as np

from . import _numpy_wrappers as nwrap

pi = sage.all.pi
Integer = sage.all.Integer
matrix_class = sage.matrix.matrix0.Matrix
vector_class = sage.structure.element.Vector

def _vectorize(func):
    def vfunc(arr):
        flat_res = np.array([func(elt) for elt in arr.flatten()],
                            dtype=arr.dtype)
        return flat_res.reshape(arr.shape)
    return vfunc

def _sage_eig(mat):
    D, P = mat.eigenmatrix_right()
    return (D.diagonal(), P)

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

def sage_vector_list(v):
    v_flat = v.reshape((-1, v.shape[-1]))
    return [sage_vector(vec) for vec in v_flat]

def sage_matrix_list(A):
    mat_flat = A.reshape((-1,) + A.shape[-2:])
    return [sage_matrix(mat) for mat in mat_flat]

def guess_base_ring(arr):
    return sage_vector(arr.flatten()).base_ring()

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

def eig(A):
    if inexact_type(A.dtype):
        return np.linalg.eig(A)

    mat_flat = list(A.reshape((-1,) + A.shape[-2:]))

    eig_res = [_sage_eig(sage_matrix(mat)) for mat in mat_flat]
    eigvals, eigvecs = zip(*eig_res)

    return (np.array(eigvals).reshape(A.shape[:-2] + (A.shape[-1],)),
            np.array(eigvecs).reshape(A.shape))

def eigh(A):
    if inexact_type(A.dtype):
        return np.linalg.eigh(A)

    return eig(A)

def det(A):
    return apply_matrix_func(
        A, np.linalg.det, lambda M: M.det(),
        expected_shape=A.shape[:-2]
    )
