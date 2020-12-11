import numpy as np
import scipy

def rotation_matrix(angle):
    return np.array([[np.cos(angle), -1*np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])

def dimension_to_axis(array, dimension, axis):
    dim_index = axis
    if array.shape[axis] != dimension:
        dim_index = self.shape.index(dimension)
        return array.swapaxes(axis, dim_index), dim_index

    return array, axis

def permutation_matrix(permutation):
    n = len(permutation)
    p_mat = np.zeros((n, n))
    for i,j in enumerate(permutation):
        p_mat[i,j] = 1.

    return p_mat

def circle_angles(center, coords):
    xs = (coords - np.expand_dims(center, axis=-2))[..., 0]
    ys = (coords - np.expand_dims(center, axis=-2))[..., 1]

    return np.arctan2(ys, xs)

def apply_bilinear(v1, v2, bilinear_form=None):
    #assuming v1 and v2 are the same shape, find all the products
    #<v1_i, v2_i>

    if bilinear_form is None:
        bilinear_form = np.identity(v1.shape[-1])

    return ((v1 @ bilinear_form) * v2).sum(-1)

def normsq(vectors, bilinear_form=None):
    return apply_bilinear(vectors, vectors, bilinear_form)


def normalize(vectors, bilinear_form=None):
    norms = normsq(vectors, bilinear_form)

    return (vectors.T / np.sqrt(np.abs(norms.T))).T

def short_arc(thetas):
    thetas[thetas < 0] += 2 * np.pi
    thetas.sort(axis=-1)

    thetas[thetas[:,1] - thetas[:,0] > np.pi] = np.flip(
        thetas[thetas[:,1] - thetas[:,0] > np.pi], axis=-1
    )

    return thetas

def sphere_inversion(points):
    return (points.T / (normsq(points)).T).T

def swap_matrix(i, j, n):
    permutation = list(range(n))
    permutation[i] = j
    permutation[j] = i
    return permutation_matrix(permutation)

def projection(v1, v2, bilinear_form):
    #project v1 onto v2

    return (v2.T *
            apply_bilinear(v1, v2, bilinear_form).T /
            normsq(v2, bilinear_form).T).T

def indefinite_orthogonalize(form, matrices):
    if len(matrices.shape) < 2:
        return normalize(matrices, form)

    n, m = matrices.shape[-2:]

    result = np.zeros_like(matrices)

    #Gram-Schmidt for an indefinite bilinear form.
    #we're using a python for loop, but only over dimension^2 which
    #is probably small

    for i in range(n):
        row = matrices[..., i, :]
        for j in range(i):
            row -= projection(row, result[..., j, :], form)
        result[..., i, :] = row

    return normalize(result, form)

def find_isometry(form, partial_map, force_oriented=False):
    #find a form-preserving matrix agreeing with a specified map on
    #the flag defined by the standard basis

    orth_partial = indefinite_orthogonalize(form, partial_map)
    if len(orth_partial.shape) < 2:
        orth_partial = np.expand_dims(orth_partial, axis=0)

    _, _, vh = np.linalg.svd(orth_partial @ form)

    kernel = vh[..., orth_partial.shape[-2]:, :]

    orth_kernel = indefinite_orthogonalize(form, kernel)
    iso = np.concatenate([orth_partial, orth_kernel], axis=-2)

    if force_oriented:
        iso = make_orientation_preserving(iso)

    return iso

def make_orientation_preserving(matrix):
    preserved = matrix.copy()
    preserved[reversing_index, ..., -1, :] *= -1
    return preserved

def pairwise_matrix_product(array1, array2):
    """do matrix multiplication where we treat array1 and array2 as
    ndarrays of matrices.
    """

    reshape1, reshape2 = array1, array2

    excess1 = reshape1.ndim - 2
    excess2 = reshape2.ndim - 2

    if excess1 > 0:
        reshape1 = np.expand_dims(array1,
                                  axis=tuple(range(excess1, excess1 + excess2)))

    if excess2 > 0:
        reshape2 = np.expand_dims(array2, axis=tuple(range(excess1)))

    product = reshape1 @ reshape2

    return product
