"""Provide utility functions used by the various geometry tools in
this package.

"""
import numpy as np

from scipy.optimize import linprog

def rotation_matrix(angle):
    r"""Get a 2x2 rotation matrix rotating counterclockwise by the
    specified angle.

    Parameters
    ----------
    angle : float
        angle to rotate by

    Returns
    --------
    ndarray
        numpy array of shape (2,2) of the form
        \(\begin{pmatrix}\cos \theta & -\sin \theta\\\sin\theta & \cos \theta
        \end{pmatrix}\)

    """
    return np.array([[np.cos(angle), -1*np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])

def permutation_matrix(permutation):
    """Return a permutation matrix representing the given permutation.

    Parameters
    -----------
    permutation: iterable
        a sequence of n numbers (indices), specifying a permutation of
        (1, ... n).

    Returns
    ---------
    ndarray
        square 2-dimensional array, giving a permutation matrix.

    """
    n = len(permutation)
    p_mat = np.zeros((n, n))
    for i,j in enumerate(permutation):
        p_mat[i,j] = 1.

    return p_mat

def diagonalize_form(bilinear_form, order_eigenvalues="signed", reverse=False):
    r"""Return a matrix conjugating a symmetric real bilinear form to a
    diagonal form.

    Parameters
    ----------
    bilinear_form: ndarray
        numpy array of shape (n, n) representing, a symmetric bilinear
        form in standard coordinates
    order_eigenvalues: string
        If "signed" (the default), conjugate to a diagonal bilinear
        form on R^n, whose eigenvectors are ordered in order of
        increasing eigenvalue.

        If "minkowski", conjugate to a diagonal bilinear form whose
        basis vectors are ordered so that lightlike basis vectors come
        first, followed by spacelike basis vectors, followed by
        timelike basis vectors. (lightlike vectors pair to a negative
        value under the form, spacelike vectors pair to a positive
        value, and timelike vectors pair to zero).
    reverse : bool
        if True, reverse the order of the basis vectors for the
        diagonal bilinear form (from the order specified by
        `order_eigenvalues`).

    Returns
    -------
    ndarray
        numpy array of shape (n, n), representing a coordinate change
        taking the given bilinear form to a diagonal form. If \(B\) is
        the matrix given by `bilinear_form`, and \(D\) is a diagonal
        matrix with the same signature as \(B\), then this function
        returns a matrix \(M\) such that \(M^TDM = B\).

    """
    n, _ = bilinear_form.shape

    eigs, U = np.linalg.eigh(bilinear_form)
    D = np.diag(1 / np.sqrt(np.abs(eigs)))


    perm = np.identity(n)

    if order_eigenvalues:
        if order_eigenvalues == "signed":
            order = np.argsort(eigs)
        if order_eigenvalues == "minkowski":
            negative = np.count_nonzero(eigs < 0)
            positive = np.count_nonzero(eigs > 0)

            if negative <= positive:
                order = np.concatenate(
                    ((eigs < 0).nonzero()[0],
                     (eigs > 0).nonzero()[0],
                     (eigs == 0).nonzero()[0])
                )
            else:
                order = np.concatenate(
                    ((eigs > 0).nonzero()[0],
                     (eigs < 0).nonzero()[0],
                     (eigs == 0).nonzero()[0])
                )
        if reverse:
            order = np.flip(order)

        perm = permutation_matrix(order)

    W = U @ D @ np.linalg.inv(perm)

    return W

def circle_angles(center, coords):
    """Return angles relative to the center of a circle.

    Parameters
    -----------
    center: ndarray
        numpy array with shape `(..., 2)` representing x,y coordinates
        the centers of some circles.
    coords: ndarray
        numpy array with shape `(..., 2)` representing x,y coordinates
        of some points.

    Returns
    --------
    ndarray
        angles (relative to x-axis) of each of the pair of points
        specified by `coords.`

    """
    xs = (coords - np.expand_dims(center, axis=-2))[..., 0]
    ys = (coords - np.expand_dims(center, axis=-2))[..., 1]

    return np.arctan2(ys, xs)

def apply_bilinear(v1, v2, bilinear_form=None):
    """Apply a bilinar form to a pair of arrays of vectors.

    Parameters
    ----------
    v1, v2: ndarray
        ndarrays of shape `(..., n)` giving arrays of vectors.
    bilinear_form: ndarray
        ndarray of shape `(n, n)` specifying a matrix representing a
        symmetric bilinear form to use to pair `v1` and `v2`. If
        `None`, use the standard (Euclidean) bilinear form on R^n.

    Returns
    -------
    ndarray
        ndarray representing the result of pairing the vectors in `v1`
        with `v2`.

    """

    if bilinear_form is None:
        bilinear_form = np.identity(v1.shape[-1])

    return ((v1 @ bilinear_form) * v2).sum(-1)

def normsq(vectors, bilinear_form=None):
    """Evaluate the norm squared of an array of vectors, with respect to a bilinear form.

    Shorthand for `apply_bilinear(vectors, vectors, bilinear_form)`.
    """
    return apply_bilinear(vectors, vectors, bilinear_form)


def normalize(vectors, bilinear_form=None):
    """Normalize an array of vectors, with respect to a bilinear form.

    Parameters
    ----------
    vectors: ndarray
        ndarray of shape `(..., n)`
    bilinear_form: ndarray or `None`
        bilinear form to use to evaluate the norms in `vectors`. If
        `None`, use the standard (Euclidean) bilinear form on R^n.

    Returns
    -------
    ndarray
        ndarray with the same shape as `vectors`. Each vector in this
        array has norm either +1 or -1, depending on whether the
        original vector had positive or negative square-norm (with
        respect to the given bilinear form).

    """
    norms = normsq(vectors, bilinear_form)
    return vectors / np.sqrt(np.abs(np.expand_dims(norms, axis=-1)))

def short_arc(thetas):
    """Reorder angles so that the counterclockwise arc between them is
    shorter than the clockwise angle.

    Parameters
    ------------
    thetas: ndarray
        numpy array of shape (..., 2), giving ordered pairs of angles
        in the range (-2pi, 2pi)

    Returns
    --------
    ndarray
        numpy array with the same shape as `thetas`. The ordered pairs
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
    ------------
    thetas: ndarray
        numpy array of shape (..., 2), giving ordered pairs of angles.

    Returns
    --------
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
    --------
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

    This realizes the map \(v \mapsto v / ||v||^2\).

    Parameters
    ----------
    points : ndarray
        Array of shape `(..., n)` giving a set of points in R^n

    Returns
    --------
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
    --------
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
    \[
    v_2 \cdot \langle v_1, v_2 \rangle / \langle v_2, v_2 \rangle,
    \]
    where \(\langle \cdot, \cdot \rangle\) is the pairing determined by
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
    --------
    w : ndarray
        vector in R^n giving the projection of `v1` onto `v2`.

    """

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

    Parameters
    -----------------
    form : ndarray of shape `(n, n)`
        the bilinear map to preserve

    partial_map : ndarray of shape `(..., k, n)`
        array representing the images of the first k standard basis
        vectors (row vectors)

    force_oriented : boolean
        whether we should apply a reflection to force the resulting
        map to be orientation-preserving.

    Returns
    -------------
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

    _, _, vh = np.linalg.svd(orth_partial @ form)

    kernel = vh[..., orth_partial.shape[-2]:, :]

    orth_kernel = indefinite_orthogonalize(form, kernel)
    iso = np.concatenate([orth_partial, orth_kernel], axis=-2)

    if force_oriented:
        iso = make_orientation_preserving(iso)

    return iso

def find_definite_isometry(partial_map, force_oriented=False):
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

    `([object axes], [excess axes], [unit axes])`,

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

def find_positive_functional(positive_points):
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

def invert_gen(generator):
    if generator.lower() == generator:
        return generator.upper()
    return generator.lower()

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
    ------------
    p1, p2, p3 : ndarray of shape `(..., 2)`
        Euclidean coordinates of three points in the plane (or three
        arrays of points)

    Returns
    --------
    tuple
        tuple of the form `(center, radius)`, where `center` is an
        ndarray containing Euclidean coordinates of the center of the
        determined circle, and `radius` is either a float or an
        ndarray containing the radius of the determined circle.

    """
    t_p1 = p1 - p3
    t_p2 = p2 - p3

    x1 = t_p1[..., 0]
    y1 = t_p1[..., 1]

    x2 = t_p2[..., 0]
    y2 = t_p2[..., 1]

    r_sq = np.stack([x1**2 + y1**2, x2**2 + y2**2], axis=-1)
    r_sq = np.expand_dims(r_sq, axis=-2)

    mats = np.stack([t_p1, t_p2], axis=-1)

    # we'll act on the right
    t_ctrs = np.squeeze(r_sq @ np.linalg.inv(mats), axis=-2)/2.

    radii = np.linalg.norm(t_ctrs, axis=-1)

    return (t_ctrs + p3, radii)

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
    ---------
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
