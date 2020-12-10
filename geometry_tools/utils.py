import numpy as np
import scipy

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

def normsq(vectors, bilinear_form=None):
    vecs = vectors
    if bilinear_form is None:
        bilinear_form = np.identity(vectors.shape[-1])

    if len(vectors.shape) < 2:
        vecs = np.expand_dims(vectors, axis=0)

    return np.diagonal(vecs @ bilinear_form @ vecs.swapaxes(-1,-2), axis1=-2, axis2=-1)

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

def indefinite_orthogonalize(form, matrix):
    #this is really inefficient for large arrays, and probably unstable.
    n, m = matrix.shape
    if m < 1:
        return
    orthogonalized = np.array(matrix[:,0]).reshape(n,1)
    for i in range(1, m):
        complement = scipy.linalg.null_space(orthogonalized.T @ form)
        orthogonalized = np.column_stack([orthogonalized, complement[:,0]])

    norm_sq = 1./np.diagonal(orthogonalized.T @ form @ orthogonalized)

    return orthogonalized @ np.diag(np.sqrt(np.abs(norm_sq)))
