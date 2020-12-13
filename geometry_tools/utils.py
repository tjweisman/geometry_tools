"""geometry_tools.utils

Module to provide utility functions used by the various geometry tools
in this package.

"""

import numpy as np
import scipy

def rotation_matrix(angle):
    """2x2 rotation matrix rotating counterclockwise by the specified
    angle.

    """
    return np.array([[np.cos(angle), -1*np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])

def dimension_to_axis(array, dimension, axis):
    """Return an array where the given axis has length dimension, if
    possible.
    """
    dim_index = axis
    if array.shape[axis] != dimension:
        dim_index = self.shape.index(dimension)
        return array.swapaxes(axis, dim_index), dim_index

    return array, axis

def permutation_matrix(permutation):
    """Return an nxn permutation matrix representing the given
    permutation.

    permutation is a sequence of n numbers (indices).

    """
    n = len(permutation)
    p_mat = np.zeros((n, n))
    for i,j in enumerate(permutation):
        p_mat[i,j] = 1.

    return p_mat

def circle_angles(center, coords):
    """return angles relative to the center of a circle.

    center is an ndarray of shape (..., 2) representing x,y
    coordinates the centers of some circles.

    coords is an ndarray of shape (..., 2,2) representing x,y
    coordinates of a pair of points.

    Return: ndarray of shape (..., 2) representing angles (relative to
    x-axis) of each of the pair of points specified by coords.

    """
    xs = (coords - np.expand_dims(center, axis=-2))[..., 0]
    ys = (coords - np.expand_dims(center, axis=-2))[..., 1]

    return np.arctan2(ys, xs)

def apply_bilinear(v1, v2, bilinear_form=None):
    """apply a bilinar form to a pair of arrays of vectors.

    if v1 and v2 are ndarrays of shape (..., n) and (..., n), apply
    the bilinear form elementwise to them, using standard broadcasting
    rules.

    """

    if bilinear_form is None:
        bilinear_form = np.identity(v1.shape[-1])

    return ((v1 @ bilinear_form) * v2).sum(-1)

def normsq(vectors, bilinear_form=None):
    """norm of an ndarray of vectors"""
    return apply_bilinear(vectors, vectors, bilinear_form)


def normalize(vectors, bilinear_form=None):
    norms = normsq(vectors, bilinear_form)

    return (vectors.T / np.sqrt(np.abs(norms.T))).T

def short_arc(thetas):
    """reorder angles so that the counterclockwise arc between them is
    shorter than the clockwise angle.

    thetas: ndarray of pairs of angles in the range (-2pi, 2pi)

    return: ndarray of pairs of angles. a subset of the pairs in
    thetas have been swapped.

    """
    thetas[thetas < 0] += 2 * np.pi
    thetas.sort(axis=-1)

    thetas[thetas[...,1] - thetas[...,0] > np.pi] = np.flip(
        thetas[thetas[...,1] - thetas[...,0] > np.pi], axis=-1
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
    """project v1 onto v2, with respect to a (hopefully nondegenerate)
    bilinear form"""

    return (v2.T *
            apply_bilinear(v1, v2, bilinear_form).T /
            normsq(v2, bilinear_form).T).T

def indefinite_orthogonalize(form, matrices):
    """apply the Gram-Schmidt algorithm, but for a possibly indefinite
    bilinear form."""
    if len(matrices.shape) < 2:
        return normalize(matrices, form)

    n, m = matrices.shape[-2:]

    result = np.zeros_like(matrices)

    #we're using a python for loop, but only over dimension^2 which
    #is probably small

    for i in range(n):
        row = matrices[..., i, :]
        for j in range(i):
            row -= projection(row, result[..., j, :], form)
        result[..., i, :] = row

    return normalize(result, form)

def find_isometry(form, partial_map, force_oriented=False):
    """find a form-preserving matrix agreeing with a specified map on
    the flag defined by the standard basis.

    form: the bilinear map to preserve

    partial_map: array of shape (..., k , n), representing the images
    of the first k standard basis vectors (row vectors)

    force_oriented: whether we should apply a reflection to force the
    resulting map to be orientation-preserving.

    return: ndarray of shape (..., n, n) representing an array of
    matrices whose rows and columns are "orthonormal" with respect to
    the bilinear form (may have norm -1).

    For all j < k, the subspace spanned by the first j standard basis
    vectors is sent to the subspace spanned by the first j rows of the
    result.

    """

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
    """apply a reflection to the last row of matrix to make it orientation
    preserving.

    if matrix is already orientation preserving, do nothing.

    """
    preserved = matrix.copy()
    preserved[np.linalg.det(preserved) < 0, -1, :] *= -1
    return preserved

def expand_unit_axes(array, unit_axes, new_axes):
    """expand the last axes of an array to make it suitable for ndarray
    elementwise multiplication.

    array is an ndarray whose last unit_axes axes are a "unit"
    (treated as a single unit for elementwise multiplication). Its
    shape is ([object axes], [unit axes]), where [object axes] and
    [unit axes] are tuples.

    This function returns an array of shape
    ([object axes], 1, ..., 1, [unit axes]),
    where the number of 1's is either (new_axes - unit_axes) or 0.

    """
    if new_axes <= unit_axes:
        return array

    return np.expand_dims(array.T, axis=tuple(range(unit_axes, new_axes))).T

def squeeze_excess(array, unit_axes, other_unit_axes):
    """squeeze excess axes from an ndarray of arrays with unit_axes axes.

    This undoes expand_unit_axes. If array is an ndarray of shape

    ([object axes], [excess axes], [unit axes]),

    where [unit axes] has length unit_axes, and [excess axes] has
    length other_unit_axes - unit_axes, this function reshapes the
    array by squeezing out all the 1's in [excess axes].

    """
    squeezable = np.array(array.T.shape[unit_axes:other_unit_axes])
    (to_squeeze,) = np.nonzero(squeezable == 1)
    to_squeeze += unit_axes

    return np.squeeze(array.T, axis=tuple(to_squeeze)).T

def matrix_product(array1, array2, unit_axis_1=2, unit_axis_2=2,
                   broadcast="elementwise"):
    """do ndarray multiplication, where we treat array1 and array2 as
    ndarrays of ndarrays with ndim unit_axis1 and unit_axis2.

    broadcasting is either elementwise or pairwise.
    """

    reshape1 = expand_unit_axes(array1, unit_axis_1, unit_axis_2)
    reshape2 = expand_unit_axes(array2, unit_axis_2, unit_axis_1)

    if broadcast == "pairwise" or broadcast == "pairwise_reversed":
        large_axes = max(unit_axis_1, unit_axis_2)

        excess1 = reshape1.ndim - large_axes
        excess2 = reshape2.ndim - large_axes

        if excess1 > 0:
            if broadcast == "pairwise_reversed":
                reshape1 = np.expand_dims(reshape1,
                                          axis=tuple(range(excess1, excess1 + excess2)))
            else:
                reshape1 = np.expand_dims(reshape1,
                                          axis=tuple(range(excess2)))

        if excess2 > 0:
            if broadcast == "pairwise_reversed":
                reshape2 = np.expand_dims(reshape2, axis=tuple(range(excess1)))
            else:
                reshape2 = np.expand_dims(reshape2,
                                          axis=tuple(range(excess2, excess1 + excess2)))

    product = reshape1 @ reshape2

    if unit_axis_1 < unit_axis_2:
        product = squeeze_excess(product, unit_axis_1, unit_axis_2)

    return product
