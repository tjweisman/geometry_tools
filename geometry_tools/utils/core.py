import warnings
import functools

import numpy as np
from scipy.optimize import linprog

from . import numerical
from . import types

from ..base import GeometryError

try:
    import sage.all
    from . import sagewrap
    SAGE_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_AVAILABLE = False

def rotation_matrix(angle, like=None, **kwargs):
    r"""Get a 2x2 rotation matrix rotating counterclockwise by the
    specified angle.

    Parameters
    ----------
    angle : float
        angle to rotate by

    Returns
    -------
    ndarray
        numpy array of shape (2,2) of the form
        $\begin{pmatrix}\cos \theta & -\sin \theta\\\sin\theta & \cos \theta
        \end{pmatrix}$

    """
    if like is None:
        like = angle
    return array_like([[cos(angle), -1*sin(angle)],
                       [sin(angle), cos(angle)]],
                      like=like, **kwargs)

def permutation_matrix(permutation, inverse=False, **kwargs):
    """Return a permutation matrix representing the given permutation.

    Parameters
    ----------
    permutation: iterable
        a sequence of n numbers (indices), specifying a permutation of
        (1, ... n).
    inverse : bool
        if True, return the matrix for the inverse of the
        permutation specified.

    Returns
    -------
    ndarray
        square 2-dimensional array, giving a permutation matrix.

    """
    n = len(permutation)
    p_mat = zeros((n, n), **kwargs)
    one = number(1, **kwargs)

    for i,j in enumerate(permutation):
        if inverse:
            p_mat[j,i] = one
        else:
            p_mat[i,j] = one

    return p_mat

def conjugate_by_permutation(matrix, permutation, inverse=False):
    result = zeros(matrix.shape, like=matrix)

    for i, si in enumerate(np.moveaxis(permutation, -1, 0)):
        res_col = zeros(matrix.shape[:-1], like=matrix)
        i_take = np.expand_dims(si, (-2, -1))
        mat_col = np.take_along_axis(matrix, i_take, axis=-1)
        for j, sj in enumerate(np.moveaxis(permutation, -1, 0)):
            j_take = np.expand_dims(sj, (-2, -1))
            val = np.take_along_axis(mat_col, j_take, axis=-2)
            res_col[..., j] = np.squeeze(val, (-1, -2))
        result[..., i] = res_col

    return result

def permute_along_axis(matrix, permutation, axis, inverse=False):
    result = zeros(matrix.shape, like=matrix).swapaxes(-1, axis)
    for i, si in enumerate(np.moveaxis(permutation, -1, 0)):
        p_ind = np.expand_dims(si, (-2, -1))
        if inverse:
            values = np.take_along_axis(matrix.swapaxes(-1, axis), p_ind, axis=-1)
            result[..., i] = values.squeeze(axis=-1)
        else:
            values = np.expand_dims(matrix.swapaxes(-1, axis)[..., i], -1)
            np.put_along_axis(result, p_ind, values, axis=-1)

    return result.swapaxes(-1, axis)

def construct_diagonal(diags):
    n = diags.shape[-1]
    result = zeros(diags.shape[:-1] + (n * n,), like=diags)
    result[..., ::n+1] = diags
    return result.reshape(diags.shape[:-1] + (n, n))

def diagonalize_form(bilinear_form,
                     order_eigenvalues="signed",
                     reverse=False,
                     with_inverse=True):
    r"""Return a matrix conjugating a symmetric real bilinear form to a
    diagonal form.

    Parameters
    ----------
    bilinear_form: ndarray
        numpy array of shape (n, n) representing, a symmetric bilinear
        form in standard coordinates
    order_eigenvalues: {'signed', 'minkowski'}
        If "signed" (the default), conjugate to a diagonal bilinear
        form on R^n where the basis vectors are ordered in order of
        increasing eigenvalue.

        If "minkowski", and the bilinear form has signature (p, q, r)
        (meaning that a maximal positive definite subspace has
        dimension p, a maximal negative definite subspace has
        dimension q, and r = n - p - q), then conjugate to a bilinear
        form where the basis vectors are ordered so that spacelike
        (positive) vectors come first if p <= q and timelike
        (negative) basis vectors come first if q < p.
    with_inverse : bool
        If True (default), also return the inverse of the conjugating
        matrix.

        Computing an inverse matrix with exact algebraic entries in
        Sage can be slow, so this will often save time.
    reverse : bool
        if True, reverse the order of the basis vectors for the
        diagonal bilinear form (from the order specified by
        `order_eigenvalues`).

    Returns
    -------
    ndarray or (ndarray, ndarray)
        numpy array of shape (n, n), representing a coordinate change
        taking the given bilinear form to a diagonal form on which the
        standard basis vectors have square-norm +-1.

        If $B$ is the matrix given by bilinear_form, and $D$ is a
        diagonal matrix with the same signature as $B$, then this
        function returns a matrix $W$ such that $W^TBW = D$. If
        with_inverse is True, then this function returns the pair (W,
        inverse(W)).
    """

    n = bilinear_form.shape[-1]

    eigs, U = eigh(bilinear_form)
    Uinv = conjugate(U.swapaxes(-1, -2))

    Dinv = construct_diagonal(np.sqrt(np.abs(eigs)))
    D = zeros(Dinv.shape, like=Dinv)

    close_to_zero = np.isclose(Dinv.astype('float64'), 0.)
    np.divide(1, Dinv, out=D, where=~close_to_zero)

    n_eigs = eigs.astype('float64')

    W = U @ D

    if order_eigenvalues:
        if order_eigenvalues == "signed":
            order = np.argsort(n_eigs, axis=-1)
        if order_eigenvalues == "minkowski":
            sort_indices = np.zeros_like(n_eigs)
            num_positive = np.count_nonzero(n_eigs < 0, axis=-1)
            num_negative = np.count_nonzero(n_eigs > 0, axis=-1)

            flip = np.ones_like(n_eigs)
            flip[num_negative < num_positive] = -1

            sort_indices[n_eigs < 0] = -1 * flip[n_eigs < 0]
            sort_indices[n_eigs > 0] = flip[n_eigs > 0]
            sort_indices[n_eigs == 0] == 2

            order = np.argsort(sort_indices, axis=-1)
        if reverse:
            order = np.flip(order)

        W = permute_along_axis(W, order, axis=-1, inverse=True)

    if not with_inverse:
        return W

    Winv = Dinv @ Uinv

    if order_eigenvalues:
        Winv = permute_along_axis(Winv, order, axis=-2, inverse=True)

    return (W, Winv)

def circle_angles(center, coords):
    """Return angles relative to the center of a circle.

    Parameters
    ----------
    center: ndarray
        numpy array with shape (..., 2) representing x,y coordinates
        the centers of some circles.
    coords: ndarray
        numpy array with shape (..., 2) representing x,y coordinates
        of some points.

    Returns
    -------
    ndarray
        angles (relative to x-axis) of each of the pair of points
        specified by coords.

    """
    xs = (coords - np.expand_dims(center, axis=-2))[..., 0]
    ys = (coords - np.expand_dims(center, axis=-2))[..., 1]

    return np.arctan2(ys, xs)

def apply_bilinear(v1, v2, bilinear_form=None,
                   broadcast="elementwise"):
    """Apply a bilinear form to a pair of arrays of vectors.

    Parameters
    ----------
    v1, v2: ndarray
        ndarrays of shape (..., n) giving arrays of vectors.
    bilinear_form: ndarray
        ndarray of shape (n, n) specifying a matrix representing a
        symmetric bilinear form to use to pair v1 and v2. If
        `None`, use the standard (Euclidean) bilinear form on R^n.

    Returns
    -------
    ndarray
        ndarray representing the result of pairing the vectors in v1
        with v2.

    """

    intermed = np.expand_dims(v1, -2)

    if bilinear_form is not None:
        intermed = matrix_product(
            np.expand_dims(v1, -2), bilinear_form,
            broadcast=broadcast
        )
    prod = matrix_product(
        intermed, np.expand_dims(v2, -1)
    )

    return prod.squeeze((-1, -2))

    #return np.array(((v1 @ bilinear_form) * v2).sum(-1))

def normsq(vectors, bilinear_form=None):
    """Evaluate the norm squared of an array of vectors, with respect to a
       bilinear form.

    Equivalent to a call of apply_bilinear(vectors, vectors, bilinear_form).

    """
    return apply_bilinear(vectors, vectors, bilinear_form)

def normalize(vectors, bilinear_form=None):
    """Normalize an array of vectors, with respect to a bilinear form.

    Parameters
    ----------
    vectors: ndarray
        ndarray of shape (..., n)
    bilinear_form: ndarray or None
        bilinear form to use to evaluate the norms in vectors. If
        None, use the standard (Euclidean) bilinear form on R^n.

    Returns
    -------
    ndarray
        ndarray with the same shape as vectors. Each vector in this
        array has norm either +1 or -1, depending on whether the
        original vector had positive or negative square-norm (with
        respect to the given bilinear form).

    """
    sq_norms = normsq(vectors, bilinear_form)

    abs_norms = np.sqrt(np.abs(np.expand_dims(sq_norms, axis=-1)))

    return np.divide(vectors, abs_norms, out=vectors,
                     where=(abs_norms.astype('float64') != 0))

def short_arc(thetas):
    """Reorder angles so that the counterclockwise arc between them is
    shorter than the clockwise angle.

    Parameters
    ----------
    thetas: ndarray
        numpy array of shape (..., 2), giving ordered pairs of angles
        in the range (-2pi, 2pi)

    Returns
    -------
    ndarray
        numpy array with the same shape as thetas. The ordered pairs
        in this array are arranged so that for each pair `(a, b)`, the
        counterclockwise arc from `a` to `b` is shorter than the
        counterclockwise arc from `b` to `a`.

    """
    shifted_thetas = np.copy(thetas)

    shifted_thetas[shifted_thetas < 0] += 2 * np.pi
    shifted_thetas.sort(axis=-1)

    to_flip = shifted_thetas[...,1] - shifted_thetas[...,0] > np.pi
    shifted_thetas[to_flip] = np.flip(shifted_thetas[to_flip], axis=-1)

    return shifted_thetas

def right_to_left(thetas):
    """Reorder angles so that the counterclockwise arc goes right to left.

    Parameters
    ----------
    thetas: ndarray
        numpy array of shape (..., 2), giving ordered pairs of angles.

    Returns
    -------
    ndarray
        numpy array with the same shape as `thetas`. The ordered pairs
        in this array are arranged so that for each pair `(a, b)`, the
        cosine of `b` is at most the cosine of `a`.

    """
    flipped_thetas = np.copy(thetas)

    to_flip = np.cos(thetas[..., 0]) < np.cos(thetas[..., 1])
    flipped_thetas[to_flip] = np.flip(thetas[to_flip], axis=-1)

    return flipped_thetas


def arc_include(thetas, reference_theta):
    """Reorder angles so that the counterclockwise arc between them always
    includes some reference point on the circle.

    Parameters
    ----------
    thetas : ndarray(float)
        pairs of angles in the range (-2pi, 2pi).
    reference_theta: ndarray(float)
         angles in the range (-2pi, 2pi)

    Returns
    -------
    theta_p : ndarray(float)
        pairs of angles of the same shape as `thetas`, so that
        `reference_theta` lies in the counterclockwise angle between
        `theta_p[..., 0]` and `theta_p[..., 1]`.

    """

    s_thetas = np.copy(thetas)
    s_theta1 = thetas[..., 1] - thetas[..., 0]
    s_reference = np.expand_dims(reference_theta - thetas[..., 0],
                                 axis=-1)

    s_theta1[s_theta1 < 0] += 2 * np.pi
    s_reference[s_reference < 0] += 2 * np.pi

    to_swap = (s_theta1 < s_reference[..., 0])

    s_thetas[to_swap] = np.flip(s_thetas[to_swap], axis=-1)
    return s_thetas

def sphere_inversion(points):
    r"""Apply unit sphere inversion to points in R^n.

    This realizes the map $v \mapsto v / ||v||^2$.

    Parameters
    ----------
    points : ndarray
        Array of shape `(..., n)` giving a set of points in R^n

    Returns
    -------
    ndarray
        The image of the `points` array under sphere inversion.

    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return (points.T / (normsq(points)).T).T

def swap_matrix(i, j, n):
    """Return a permutation matrix representing a single transposition.

    Parameters
    ----------
    i, j : int
        Indices swapped by a transposition.
    n : int
        dimension of the permutation matrix.

    Returns
    -------
    ndarray
        Array of shape `(n, n)` giving a transposition matrix.

    """
    permutation = list(range(n))
    permutation[i] = j
    permutation[j] = i
    return permutation_matrix(permutation)

def projection(v1, v2, bilinear_form):
    r"""Return the projection of `v1` onto `v2` with respect to some
    bilinear form.

    The returned vector `w` is parallel to `v2`, and `v1 - w` is
    orthogonal to `v2` with respect to the given bilinear form.
    `w` is determined by the formula
    $$
    v_2 \cdot \langle v_1, v_2 \rangle / \langle v_2, v_2 \rangle,
    $$
    where $\langle \cdot, \cdot \rangle$ is the pairing determined by
    `bilinear_form`.

    Parameters
    ----------
    v1 : ndarray
        vector in R^n to project
    v2 : ndarray
        vector in R^n to project onto
    bilinear_form : ndarray
        array of shape `(n, n)` giving a bilinear form to use for the
        projection.

    Returns
    -------
    w : ndarray
        vector in R^n giving the projection of `v1` onto `v2`.

    """

    return (v2.T *
            apply_bilinear(v1, v2, bilinear_form).T /
            normsq(v2, bilinear_form).T).T

def orthogonal_complement(vectors, form=None, normalize="form"):
    """Find an orthogonal complement with respect to a nondegenerate (but
       possibly indefinite) bilinear form.

    Parameters
    ----------
    vectors : ndarray of shape (..., k, n)
        array of k row vectors for which to find a complement. Each
        set of k row vectors should be linearly independent.

    form : ndarray of shape `(n, n)`
        bilinear form to use to find complements. If None (the
        default), use the standard Euclidean form on R^n.

    normalize : One of {'form', 'euclidean'} or None
        How to normalize the vectors spanning the orthogonal
        complement. If 'form' (the default), attempt to return vectors
        which have unit length with respect to `form.` (Note this may
        fail if `form` is indefinite, i.e. there are nonzero null
        vectors. If 'euclidean', then use the standard Euclidean form
        on R^n to normalize the vectors.

    Returns
    -------
    result : ndarray with shape (..., n - k, n)
        array of n-k row vectors, each of which is orthogonal to all
        of the k row vectors provided (with respect to `form`).

    """
    if form is None:
        form = np.identity(vectors.shape[-1])

    kernel_basis = kernel(vectors @ form).swapaxes(-1, -2)

    if normalize == 'form':
        return indefinite_orthogonalize(form, kernel_basis)

    return kernel_basis

def indefinite_orthogonalize(form, matrices):
    """Apply the Gram-Schmidt algorithm, but for a possibly indefinite
    bilinear form.

    Parameters
    ----------
    form : ndarray of shape `(n,n)`
        bilinear form to orthogonalize with respect to

    matrices : ndarray of shape `(..., k, n)`
        array of k row vectors to orthogonalize.

    Returns
    -------
    result : ndarray
        array with the same shape as `matrices`. The last two
        dimensions of this array give matrices with mutually
        orthogonal rows, with respect to the given bilinear form.

        For all `j <= k`, the subspace spanned by the first `j` rows
        of `result` is the same as the subspace spanned by the first
        `j` rows of `matrices`.

    """
    if len(matrices.shape) < 2:
        return normalize(matrices, form)

    n, m = matrices.shape[-2:]

    result = zeros(matrices.shape,
                   like=matrices,
                   integer_type=False)

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

    Parameters
    ----------
    form : ndarray of shape `(n, n)`
        the bilinear map to preserve

    partial_map : ndarray of shape `(..., k, n)`
        array representing the images of the first k standard basis
        vectors (row vectors)

    force_oriented : boolean
        whether we should apply a reflection to force the resulting
        map to be orientation-preserving.

    Returns
    -------
    ndarray
        array of shape `(..., n, n)` representing an array of matrices
        whose rows and columns are "orthonormal" with respect to the
        bilinear form (since the form may be indefinite, "normal"
        vectors may have norm -1).

        For all `j <= k`, the subspace spanned by the first `j`
        standard basis vectors is sent to the subspace spanned by the
        first `j` rows of the result.

    """

    orth_partial = indefinite_orthogonalize(form, partial_map)

    if len(orth_partial.shape) < 2:
        orth_partial = np.expand_dims(orth_partial, axis=0)

    kernel_basis = kernel(orth_partial @ form).swapaxes(-1, -2)

    orth_kernel = indefinite_orthogonalize(form, kernel_basis)

    iso = np.concatenate([orth_partial, orth_kernel], axis=-2)

    if force_oriented:
        iso = make_orientation_preserving(iso)

    return iso

def find_definite_isometry(partial_map, force_oriented=False):
    """find an orthogonal matrix agreeing with a specified map on
    the flag defined by the standard basis.

    Parameters
    ----------
    partial_map : ndarray of shape `(..., k, n)`
        array representing the images of the first k standard basis
        vectors (row vectors)

    force_oriented : boolean
        whether we should apply a reflection to force the resulting
        map to be orientation-preserving.

    Returns
    -------
    ndarray
        array of shape `(..., n, n)` representing an array of matrices
        with orthonormal rows and columns.

        For all `j <= k`, the subspace spanned by the first `j`
        standard basis vectors is sent to the subspace spanned by the
        first `j` rows of the result.

    """
    pmap = np.array(partial_map)
    if len(pmap.shape) < 2:
        pmap = pmap.reshape((len(pmap), 1))
    h, w = pmap.shape[-2:]
    n = max(h, w)
    if w > h:
        mat = np.concatenate([pmap.swapaxes(-1, -2),
                             np.identity(n)], axis=-1)
    else:
        mat = np.concatenate([pmap, np.identity(n)], axis=-1)

    q, r = np.linalg.qr(mat)

    iso = np.sign(r[..., 0,0]) * q

    if force_oriented:
        iso = make_orientation_preserving(iso)

    return iso

def make_orientation_preserving(matrix):
    """apply a reflection to the last row of matrix to make it orientation
    preserving.

    if matrix is already orientation preserving, do nothing.

    Parameters
    ----------
    matrix : ndarray of shape `(..., n, n)`
        ndarray of linear maps

    Returns
    -------
    result : ndarray
        array with the same shape as `matrices`, representating an
        ndarray of linear maps. If `A` is an orientation-reversing
        matrix in `matrices`, then the corresponding matrix in
        `result` has its last row negated.
    """
    preserved = matrix.copy()
    preserved[det(preserved) < 0, -1, :] *= -1
    return preserved

def expand_unit_axes(array, unit_axes, new_axes):
    """expand the last axes of an array to make it suitable for ndarray
    pairwise multiplication.

    `array` is an ndarray, viewed as an array of arrays, each of which
    has `unit_axes` axes. That is, its shape decomposes into a pair of
    tuples `([axes 1], [axes 2])`, where `[axes 2]` is a tuple of
    length `unit_axes`.

    Parameters
    ----------
    array : ndarray
        ndarray to expand
    unit_axes : int
        number of axes of `array` to treat as a "unit"
    new_axes : int
        number of axes to add to the array

    Returns
    -------
    ndarray
        ndarray of shape `([object axes], 1, ..., 1, [unit axes])`,
        where the number of 1's is either `new_axes - unit_axes`, or 0
        if this is negative.

    """
    if new_axes <= unit_axes:
        return array

    return np.expand_dims(array.T, axis=tuple(range(unit_axes, new_axes))).T

def squeeze_excess(array, unit_axes, other_unit_axes):
    """Squeeze all excess axes from an ndarray of arrays with unit_axes axes.

    This undoes expand_unit_axes.

    Parameters
    ----------
    array : ndarray
        ndarray of shape
        `([object axes], [excess axes], [unit axes])`, where `[unit axes]`
        is a tuple of length `unit_axes`, and `[excess axes]` is a
        tuple of length `other_unit_axes - unit_axes`.
    unit_axes : int
        number of axes to view as "units" in `array`. That is, `array`
        is viewed as an ndarray of arrays each with `unit_axes` axes.
    other_unit_axes : int
        axes to avoid squeezing when we reshape the array.

    Returns
    -------
    ndarray
        Reshaped array with certain length-1 axes removed. If the
        input array has shape
        `([object axes], [excess axes], [unit axes])`,
        squeeze out all the ones in `[excess axes]`.

    """
    squeezable = np.array(array.T.shape[unit_axes:other_unit_axes])
    (to_squeeze,) = np.nonzero(squeezable == 1)
    to_squeeze += unit_axes

    return np.squeeze(array.T, axis=tuple(to_squeeze)).T

def broadcast_match(a1, a2, unit_axes):
    """Tile a pair of arrays so that they can be broadcast against each
    other, treating the last ndims as a unit.

    It is often not necessary to call this function, since numpy
    broadcasting will handle this implicitly for many vectorized
    functions.

    Parameters
    ----------
    a1, a2 : ndarray
        arrays to tile
    unit_axes : int
        number of axes to treat as a unit when tiling

    Returns
    -------
    (u1, u2) : (ndarray, ndarray)
        Tiled versions of a1 and a2, respectively.

        If unit_axes is k, and a1 has shape (N1, ..., Ni, L1, ...,
        Lk), and a2 has shape (M1, ..., Mj, P1, ..., Pk), then u1 has
        shape (N1, ..., Ni, M1, ..., Mj, L1, ..., Lk) and u2 has shape
        (N1, ..., Ni, M1, ..., Mj, P1, ..., Pk).
    """
    c1_ndims = len(a1.shape) - unit_axes
    c2_ndims = len(a2.shape) - unit_axes

    exp1 = np.expand_dims(a1, tuple(range(c1_ndims, c1_ndims + c2_ndims)))
    exp2 = np.expand_dims(a2, tuple(range(c1_ndims)))

    tile1 = np.tile(exp1, (1,) * c1_ndims + a2.shape[:c2_ndims] + (1,) * unit_axes)
    tile2 = np.tile(exp2, a1.shape[:c1_ndims] + (1,) * (c2_ndims + unit_axes))

    return (tile1, tile2)

def matrix_product(array1, array2, unit_axis_1=2, unit_axis_2=2,
                   broadcast="elementwise"):
    """Multiply two ndarrays of ndarrays together.

    Each array in the input is viewed as an ndarray of smaller
    ndarrays with a specified number of axes. The shapes of these
    smaller arrays must broadcast against each other (i.e. be
    compatible with the numpy `@` operator).

    For the dimensions of the outer ndarray, the behavior of this
    function depends on the broadcast rule provided by the `broadcast`
    keyword.

    Parameters
    ----------
    array1, array2 : ndarray
        ndarrays of ndarrays to multiply together
    unit_axis_1, unit_axis_2 : int
        Each of `array1` and `array2` is viewed as an array with
        `unit_axis_1` and `unit_axis_2` ndims, respectively. By
        default both are set to 2, meaning this function multiplies a
        pair of ndarrays of matrices (2-dim arrays).
    broadcast : {'elementwise', 'pairwise', 'pairwise_reversed'}
        broadcast rule to use when multiplying arrays.

        If the broadcast rule is 'elementwise' (the default), assume that
        the shape of one outer array broadcasts against the shape of
        the other outer array, and multiply with the same rules as the
        numpy `@` operator.

        If 'pairwise', multiply every element in the first outer array
        by every element in the second outer array, and return a new
        array of arrays with expanded axes. If 'pairwise_reversed', do
        the same thing, but use the axes of the second array first in
        the result.

    Returns
    -------
    result : ndarray

        result of matrix multiplication.

        If `array1` and `array2` have shapes
        `([outer ndims 1], [inner ndims 1])` and
        `([outer ndims 2], [inner ndims 2])` (where
        `[inner ndims]` are tuples with length `unit_axis_1` and
        `unit_axis_2`, respectively) then:

        - if `broadcast` is 'elementwise', `result` has shape
          `([result ndims], [product ndims])`, where `[result ndims]`
          is the result of broadcasting the outer ndims against each
          other, and `[product ndims]` is the result of broadcasting
          the inner ndims against each other.

        - if `broadcast` is 'pairwise', result has shape `([outer
          ndims 1], [outer ndims 2], [product ndims])`.

        - if `broadcast` is 'pairwise_reversed', result has shape
          `([outer ndims 2], [outer ndims 1], [product ndims])`

    """

    reshape1 = expand_unit_axes(array1, unit_axis_1, unit_axis_2)
    reshape2 = expand_unit_axes(array2, unit_axis_2, unit_axis_1)

    if broadcast == "pairwise" or broadcast == "pairwise_reversed":
        large_axes = max(unit_axis_1, unit_axis_2)

        excess1 = reshape1.ndim - large_axes
        excess2 = reshape2.ndim - large_axes

        if excess1 > 0:
            if broadcast == "pairwise":
                reshape1 = np.expand_dims(reshape1,
                                          axis=tuple(range(excess1, excess1 + excess2)))
            else:
                reshape1 = np.expand_dims(reshape1,
                                          axis=tuple(range(excess2)))

        if excess2 > 0:
            if broadcast == "pairwise":
                reshape2 = np.expand_dims(reshape2, axis=tuple(range(excess1)))
            else:
                reshape2 = np.expand_dims(reshape2,
                                          axis=tuple(range(excess2, excess1 + excess2)))

    product = reshape1 @ reshape2

    if unit_axis_1 < unit_axis_2:
        product = squeeze_excess(product, unit_axis_1, unit_axis_2)

    return product

def find_positive_functional(positive_points):
    """Find a dual vector which evaluates to a positive real on a given
       array of vectors, using scipy's linear programming routines.

    Parameters
    ----------
    positive_points : ndarray
        array of vectors to find a dual vector for. If this is a
        2-dimensional array, then find a single dual vector which is
        positive when paired with every row of the array.

        If the array has shape (..., k, n), then find an array of dual
        vectors such that the vectors in the array pair positively
        with the corresponding k vectors in positive_points.

    Returns
    -------
    duals : ndarray
        array of dual vectors. If positive_points has shape (d1, ...,
        dj, k, n), then `duals` has shape (d1, ..., dj, n).

    """
    dim = positive_points.shape[-1]
    codim = positive_points.shape[-2]

    functionals = np.zeros(positive_points.shape[:-2] +
                           (positive_points.shape[-1],))

    for ind in np.ndindex(positive_points.shape[:-2]):
        res = linprog(
            np.zeros(dim),
            -1 * positive_points[ind],
            -1 * np.ones(codim),
            bounds=(None, None))

        if not res.success:
            return None

        functionals[ind] = res.x

    return normalize(functionals)

def first_sign_switch(array):
    signs = np.sign(array)
    row_i, col_i = np.nonzero(signs != np.expand_dims(signs[..., 0], axis=-1))
    _, init_inds = np.unique(row_i, return_index=True)
    return col_i[init_inds]

def circle_through(p1, p2, p3):
    """Get the unique circle passing through three points in the plane.

    This does NOT check for colinearity and will just return nans in
    that case.

    Parameters
    ----------
    p1, p2, p3 : ndarray of shape `(..., 2)`
        Euclidean coordinates of three points in the plane (or three
        arrays of points)

    Returns
    -------
    tuple
        tuple of the form `(center, radius)`, where `center` is an
        ndarray containing Euclidean coordinates of the center of the
        determined circle, and `radius` is either a float or an
        ndarray containing the radius of the determined circle.

    """
    return sphere_through(np.stack([p1, p2, p3], axis=-2))

def sphere_through(points):
    """Get the unique k-sphere passing through k+2 points in R^(k+1).

    This does NOT check for linear dependence, and will just return
    nans in that case.

    Parameters
    ----------
    points : ndarray of shape `(..., k)`
        Euclidean coordinates of k points in R^(k+1)

    Returns
    -------
    tuple
        tuple of the form `(center, radius)`, where `center` is an
        ndarray containing Euclidean coordinates of the center of the
        determined spheres, and `radius` is either an ndarray
        containing the radius of the determined spheres.
    """
    k = points.shape[-2]
    if points.shape[-1] != k - 1:
        raise GeometryError(
            f"Cannot construct a unique sphere through {k} points in "
            f"{points.shape[-1]} dimensions"
        )

    # translate so one of the points is at the origin
    p0 = np.expand_dims(points[..., 0, :], axis=-2)
    t_pts = points[..., 1:, :] - p0

    # solve for the translated center
    r_sq = normsq(t_pts)[..., np.newaxis, :]

    t_ctr = r_sq @ invert(t_pts.swapaxes(-1, -2)) / number(2, like=r_sq)
    radii = np.linalg.norm(t_ctr, axis=-1)

    center = np.squeeze(t_ctr + p0, axis=-2)
    radii = np.squeeze(radii, axis=-1)

    return (center, radii)


def r_to_c(real_coords):
    return real_coords[..., 0] + real_coords[..., 1]*1.0j

def c_to_r(cx_array):
    return cx_array.astype('complex').view('(2,)float')

def order_eigs(eigenvalues, eigenvectors):
    """Sort eigenvalue/eigenvector tuples by increasing modulus.

    This function accepts the eigenvalue/eigenvector data returned by
    the `np.linalg.eig` function.

    Parameters
    ----------
    eigenvalues : ndarray
        Array of shape (..., n) representing eigenvalues of some
        matrices
    eigenvectors : ndarray
        Array of shape (..., n, n) representing eigenvectors of some
        matrices (as column vectors).

    Returns
    -------
    tuple
        Tuple of the form `(eigvals, eigvecs)`, where both eigvals and
        eigvecs have the same data as the input arrays, but arranged
        so that eigenvalues and eigenvectors are in increasing order
        of modulus.

    """
    eigorder = np.argsort(np.abs(eigenvalues), axis=-1)
    eigvecs = np.take_along_axis(eigenvectors, np.expand_dims(eigorder, axis=-2), axis=-1)
    eigvals = np.take_along_axis(eigenvalues, eigorder, axis=-1)

    return eigvals, eigvecs

def affine_disks_contain(cout, rout, cin, rin,
                         broadcast="elementwise"):
    if broadcast == "pairwise":
        pairwise_dists = np.linalg.norm(
            np.expand_dims(cout, 0) - np.expand_dims(cin, 1),
            axis=-1
        )
        radial_diffs = np.expand_dims(rout, 0) - np.expand_dims(rin, 1)
        return pairwise_dists < radial_diffs

    return np.linalg.norm(cout - cin, axis=-1) < (rout - rin)


def disk_containments(cout, rout, cin, rin,
                      broadcast="elementwise"):
    if broadcast == "pairwise":
        pairwise_dists = np.linalg.norm(
            np.expand_dims(cout, 0) - np.expand_dims(cin, 1),
            axis=-1
        )
        radial_diff1 = np.expand_dims(rout, 0) - np.expand_dims(rin, 1)
        return (pairwise_dists < radial_diffs,
                pairwise_dists < -radial_diffs)

    dists = np.linalg.norm(cout - cin, axis=-1)
    return (dists < (rout - rin),
            dists < (rin - rout))

def disk_interactions(c1, r1, c2, r2,
                      broadcast="elementwise"):
    # WARNING: this will only work for Nx2 arrays
    if broadcast == "pairwise":
        pairwise_dists = np.linalg.norm(
            np.expand_dims(c1, 1) - np.expand_dims(c2, 0),
            axis=-1
        )
        radial_diff = np.expand_dims(r1, 1) - np.expand_dims(r2, 0)
        radial_sum = np.expand_dims(r1, 1) + np.expand_dims(r2, 0)
        return (pairwise_dists < radial_diff,
                pairwise_dists < -radial_diff,
                pairwise_dists < radial_sum)

    dists = np.linalg.norm(c1 - c2, axis=-1)

    return (dists < (r1 - r2),
            dists < (r2 - r1),
            dists < (r2 + r1))

def indefinite_form(p, q, neg_first=True, **kwargs):
    n = p + q

    mult = 1
    if neg_first:
        mult = -1

    form_mat = zeros((n,n), **kwargs)
    form_mat[:p, :p] = mult * identity(p, **kwargs)
    form_mat[p:, p:] = -mult * identity(q, **kwargs)

    return form_mat

def symplectic_form(n, **kwargs):
    if n != n % 2:
        raise ValueError("Cannot construct a symplectic form in odd dimensions.")

    m = n / 2

    form_mat = zeros((n, n), **kwargs)
    form_mat[:m, m:] = -identity(m, **kwargs)
    form_mat[m:, :m] = identity(m, **kwargs)

    return form_mat

def symmetric_part(bilinear_form):
    return (bilinear_form + bilinear_form.swapaxes(-1, -2)) / 2

def antisymmetric_part(bilinear_form):
    return (bilinear_form - bilinear_form.swapaxes(-1, -2)) / 2

def matrix_func(func):
    # get sage_func now, so if there's a name issue we'll throw an
    # error when the wrapped function is defined (rather than when
    # it's called)
    if SAGE_AVAILABLE:
        sage_func = getattr(sagewrap, func.__name__)

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if not SAGE_AVAILABLE:
            return func(*args, **kwargs)

        return sage_func(*args, **kwargs)

    return wrapped

@matrix_func
def invert(mat):
    return np.linalg.inv(mat)

@matrix_func
def kernel(mat):
    return numerical.svd_kernel(mat)

@matrix_func
def eig(mat):
    return np.linalg.eig(mat)

def eigh(mat):
    if not SAGE_AVAILABLE or types.is_linalg_type(mat):
        return np.linalg.eigh(mat)

    eigvals, eigvecs = sagewrap.eig(mat)
    orthogonalized = indefinite_orthogonalize(
        identity(eigvecs.shape[-1], like=eigvecs),
        eigvecs.swapaxes(-1, -2)
    ).swapaxes(-1, -2)

    return (eigvals, orthogonalized)


@matrix_func
def det(mat):
    return np.linalg.det(mat)

def check_type(base_ring=None, dtype=None, like=None,
               default_dtype='float64', integer_type=True):

    default_ring = None
    if SAGE_AVAILABLE:
        default_ring = sagewrap.Integer
        if not integer_type:
            default_ring = sagewrap.SR

    if like is not None:
        try:
            if dtype is None:
                dtype = like.dtype
        except AttributeError:
            if not types.is_linalg_type(like):
                dtype = np.dtype('O')

    if base_ring is None and dtype == np.dtype('O') and SAGE_AVAILABLE:
        base_ring = default_ring

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

    if not integer_type and np.can_cast(dtype, int):
        dtype = np.dtype('float64')

    return (base_ring, dtype)

def complex_type(**kwargs):
    base_ring, dtype = check_type(integer_type=False, **kwargs)

    if np.can_cast(dtype, 'float'):
        return base_ring, np.dtype('complex')

    return base_ring, dtype

def guess_literal_ring(data):
    if not SAGE_AVAILABLE:
        return None

    try:
        if data.dtype == np.dtype('O'):
            return sage.all.QQ
    except AttributeError:
        if not types.is_linalg_type(data):
            return sage.all.QQ

    # this is maybe not Pythonic, but it's also not a mistake.
    return None

def scalar(val):
    try:
        return val.item()
    except AttributeError:
        return val

def astype(val, dtype=None):
    if dtype is None:
        return np.array(val)

    return np.array(val).astype(dtype)

def number(val, like=None, dtype=None, base_ring=None):
    if like is not None and dtype is not None:
        raise UserWarning(
            "Passing both 'like' and 'dtype' when specifying a number;"
            " 'like' parameter is ignored."
        )

    if base_ring is not None:
        if dtype is not None:
            raise UserWarning(
                "Passing both 'base_ring' and 'dtype' when specifying a number;"
                " 'dtype' is ignored (assumed to be dtype('O'))"
            )
        dtype = np.dtype('O')

        if not SAGE_AVAILABLE:
            raise UserWarning( "Specifying base_ring when sage is not"
            "available has no effect."  )

    if not SAGE_AVAILABLE:
        return scalar(astype(val, dtype))

    if dtype is None and like is not None:
        try:
            dtype = like.dtype
        except AttributeError:
            if not types.is_linalg_type(like):
                dtype = np.dtype('O')

    if dtype == np.dtype('O'):
        if isinstance(val, int):
            # we use SR instead of Integer here, because numpy
            # sometimes silently converts sage Integers
            return sage.all.SR(val)
        if isinstance(val, float):
            return sage.all.SR(sage.all.QQ(val))

    return scalar(astype(val, dtype))

def wrap_elementary(func):
    @functools.wraps(func)
    def wrapped(*args, like=None, base_ring=None, dtype=None, **kwargs):
        if not SAGE_AVAILABLE:
            return func(*args, **kwargs)

        else:
            packaged_args = []
            arg_ring = None
            if base_ring is not None:
                arg_ring = sagewrap.SR

            for arg in args:
                arg_like = like
                if arg_like is None:
                    arg_like = arg

                packaged_args.append(
                    array_like(arg, like=arg_like,
                               base_ring=arg_ring,
                               dtype=dtype,
                               integer_type=False)
                )

            return sagewrap.change_base_ring(
                func(*packaged_args, **kwargs),
                base_ring
            )

    return wrapped

@wrap_elementary
def power(base, exp):
    return np.power(base, exp)

@wrap_elementary
def cos(arg):
    return np.cos(arg)

@wrap_elementary
def sin(arg):
    return np.sin(arg)

@wrap_elementary
def conjugate(arg):
    return np.conjugate(arg)

def real(arg):
    if not SAGE_AVAILABLE or types.is_linalg_type(arg):
        return np.real(arg)

    # np.real doesn't work well with sage since it expects the real
    # attribute to be a property, not a callable. So we need to wrap
    # sage's real function explicitly.
    return sagewrap.real(arg)

def imag(arg):
    if not SAGE_AVAILABLE or types.is_linalg_type(arg):
        return np.imag(arg)

    return sagewrap.imag(arg)

def array_like(array, like=None, base_ring=None, dtype=None,
               integer_type=False, **kwargs):

    if like is None:
        like = array

    base_ring, dtype = check_type(base_ring, dtype, like,
                                  integer_type=integer_type)

    arr = np.array(array, dtype=dtype, **kwargs)

    if base_ring is not None:
        return sagewrap.change_base_ring(arr, base_ring)
    return arr

def pi(**kwargs):
    if not SAGE_AVAILABLE:
        return np.pi

    base_ring, dtype = check_type(**kwargs)

    if base_ring is not None:
        return sagewrap.pi

    return np.pi

def unit_imag(**kwargs):
    if not SAGE_AVAILABLE:
        return 1j

    base_ring, dtype = check_type(**kwargs)

    if base_ring is not None:
        return sagewrap.I

    return 1j


def zeros(shape, base_ring=None, dtype=None, like=None,
          integer_type=True, **kwargs):

    base_ring, dtype = check_type(base_ring, dtype, like,
                                  integer_type=integer_type)

    zero_arr = np.zeros(shape, dtype=dtype, **kwargs)
    if base_ring is not None:
        return sagewrap.change_base_ring(zero_arr, base_ring)
    return zero_arr

def ones(shape, base_ring=None, dtype=None, like=None,
         integer_type=True, **kwargs):
    base_ring, dtype = check_type(base_ring, dtype, like,
                                  integer_type=integer_type)

    ones_arr = np.ones(shape, dtype=dtype, **kwargs)
    if base_ring is not None:
        return sagewrap.change_base_ring(ones_arr, base_ring)
    return ones_arr

def identity(n, base_ring=None, dtype=None, like=None,
             integer_type=True, **kwargs):
    base_ring, dtype = check_type(base_ring, dtype, like,
                                  integer_type=integer_type)

    identity_arr = np.identity(n, dtype=dtype, **kwargs)

    if base_ring is not None:
        return sagewrap.change_base_ring(identity_arr, base_ring)
    return identity_arr

def change_base_ring(array, base_ring=None,
                     rational_approx=False):
    if base_ring is None:
        return array

    if not SAGE_AVAILABLE:
        raise EnvironmentError(
            "Cannot change base ring unless sage is available."
        )

    return sagewrap.change_base_ring(array, base_ring,
                                     rational_approx=rational_approx)
