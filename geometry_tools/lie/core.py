import itertools
import numpy as np
import scipy.special

from ..base import GeometryError
from .. import utils

if utils.SAGE_AVAILABLE:
    from ..utils import sagewrap

def binom(n, k, **kwargs):
    return utils.number(scipy.special.binom(n, k),
                        **kwargs)

def _convert_vector(vector, like=None, autoconvert=True):
    is_sage_obj = False
    if like is not None and utils.SAGE_AVAILABLE:
        is_sage_obj = (isinstance(like, sagewrap.matrix_class) or
                        isinstance(like, sagewrap.vector_class))
    if (autoconvert and
        utils.SAGE_AVAILABLE and
        is_sage_obj):

        if len(vector.shape) == 1:
            return sagewrap.sage_vector(vector)

        return sagewrap.sage_vector_list(vector)

    return vector

def _convert_matrix(matrix, like=None, autoconvert=True):
    is_sage_obj = False
    if like is not None and utils.SAGE_AVAILABLE:
        is_sage_obj = (isinstance(like, sagewrap.matrix_class) or
                        isinstance(like, sagewrap.vector_class))
    if (autoconvert and
        utils.SAGE_AVAILABLE and
        is_sage_obj):
        if len(matrix.shape) == 2:
            return sagewrap.sage_matrix(matrix)
        return sagewrap.sage_matrix_list(matrix)

    return matrix

def basis_matrix(i, j, n, as_sage=False, **kwargs):
    bm = utils.zeros((n,n), **kwargs)
    one = utils.number(1, **kwargs)
    bm[i,j] = one

    if as_sage:
        return sagewrap.sage_matrix(bm)
    return bm

def sln_basis_matrix(i, j, n, as_sage=False, **kwargs):
    bm = basis_matrix(i, j, n, as_sage=as_sage, **kwargs)
    if i == j:
        one = utils.number(1, **kwargs)
        bm[n-1, n-1] = -one
    return bm


def gln_lie_algebra_coords(matrix, dtype=None,
                           autoconvert=True):

    arr = np.array(matrix, dtype=dtype)
    n = arr.shape[-1]

    coords = arr.reshape(arr.shape[:-2] + (n*n,))
    return _convert_vector(coords, like=matrix,
                           autoconvert=autoconvert)

def coords_to_gln_algebra(coord_vector, dtype=None,
                          autoconvert=True):
    arr = np.array(coord_vector, dtype=dtype)
    d = arr.shape[-1]

    n = np.sqrt(d)
    if d != int(d):
        raise GeometryError(
            "Cannot interpret a vector with non-square length as "
            "an element in a square matrix space"
        )

    mats = arr.reshape((-1, n, n))
    return _convert_matrix(mats, like=coord_vector,
                           autoconvert=autoconvert)


def sln_lie_algebra_coords(matrix, dtype=None,
                           autoconvert=True):
    """Compute an explicit isomorphism between the space of traceless
    square matrices with coefficients in a ring R and the module
    R^(n^2 - 1).

    Parameters
    ----------
    matrix : array-like
        A traceless (n x n) matrix. Must be interpretable as a numpy
        ndarray (so Sage matrices are allowed).
    dtype : dtype
        The underlying dtype of the returned vector. If None, use the
        dtype of matrix.
    autoconvert : bool
        If True, and the input matrix is a Sage matrix, then convert
        the output to a Sage vector.

    Returns
    -------
    vector : ndarray or Sage vector
        Array with shape (..., n^2 - 1).
    """
    arr = np.array(matrix, dtype=dtype)
    n = arr.shape[-1]

    coords = arr.reshape(arr.shape[:-2] + (n*n,))[..., :-1]

    return _convert_vector(coords, like=matrix,
                           autoconvert=autoconvert)

def coords_to_sln_lie_algebra(coord_vector, dtype=None,
                              autoconvert=True):

    arr = np.array(coord_vector, dtype=dtype)
    aug_arr = np.concatenate([arr, utils.zeros(arr.shape[:-1] + (1,),
                                               like=coord_vector,
                                               dtype=dtype)],
                             axis=-1)

    gln_coords = coords_to_gln_lie_algebra(aug_arr, dtype=dtype,
                                           autoconvert=False)

    gln_coords[..., -1, -1] = -np.trace(gln_coords, axis1=-1, axis2=-2)

    return _convert_matrix(gln_coords, like=coord_vector,
                           autoconvert=autoconvert)

def linear_matrix_action(linear_map, n, **kwargs):
    base_ring, dtype = utils.check_type(**kwargs)
    map_matrix = utils.zeros((n*n, n*n), base_ring, dtype)

    for i in range(n):
        for j in range(n):
            bm = basis_matrix(i, j, n, like=map_matrix)

            b_image = linear_map(bm)

            map_matrix[:, i*n + j] = gln_lie_algebra_coords(
                b_image, autoconvert=False
            )

    return map_matrix

def sln_linear_action(linear_map, n, **kwargs):
    base_ring, dtype = utils.check_type(**kwargs)
    map_matrix = utils.zeros((n**2 - 1, n**2 - 1), base_ring, dtype)

    for i in range(n):
        for j in range(n):
            if i == n - 1 and j == n - 1:
                break

            bm = sln_basis_matrix(i, j, n, like=map_matrix)

            b_image = linear_map(bm)

            map_matrix[:, i*n + j] = sln_lie_algebra_coords(
                b_image, autoconvert=False
            )

    return map_matrix

def sln_adjoint(mat, inv=None, **kwargs):
    n = mat.shape[-1]
    if inv is None:
        inv = utils.invert(mat)

    return sln_linear_action(
        lambda M: mat @ M @ inv,
        n, **kwargs
    )

def gln_adjoint(mat, inv=None, **kwargs):
    n = mat.shape[-1]
    if inv is None:
        inv = utils.invert(mat)

    return linear_matrix_action(
        lambda M: mat @ M @ inv,
        n, **kwargs
    )

def sl2_irrep(A, n):
    r"""The irreducible representation \(\mathrm{SL}(2) \to
    \mathrm{SL}(n)\), via the action on homogeneous polynomials.

    Given an element of \(\mathrm{SL}(2)\) as a 2x2 array, compute a
    matrix giving the action of this matrix on symmetric polynomials
    in elements of the standard basis \(\{e_1, e_2\}\). The (ordered)
    basis for the new matrix is given by the degree-(n-1) monomials
    \(\{e_1^{0} e_2^{n-1}, e_1^{1} e_2^{n-2}, \ldots, e_1^{n-1}e_2^{0}\}\).

    Parameters
    ----------
    A : ndarray
        Array of shape `(..., 2, 2)`, giving a matrix (or array of
        matrices) to represent.
    n : int
        Dimension of the irreducible representation.

    Returns
    -------
    result : ndarray
        Array of shape `(..., n, n)` giving the representation of
        `A` under the `dim`-dimensional irreducible representation of
        \(\mathrm{SL}(2)\).

    """

    a = A[..., 0, 0]
    b = A[..., 0, 1]
    c = A[..., 1, 0]
    d = A[..., 1, 1]

    im = utils.zeros(A.shape[:-2] +(n, n), like=A)
    r = n - 1
    for k in range(n):
        for j in range(n):
            for i in range(max(0, j - r + k), min(j+1, k+1)):
                im[..., j,k] += (binom(k, i, like=A) * binom(r - k, j - i, like=A)
                          * a**i * c**(k - i) * b**(j - i)
                          * d**(r - k - j + i))
    return im

def sln_killing_form(n, **kwargs):
    sln_dim = n*n - 1
    form = utils.zeros((sln_dim, sln_dim), **kwargs)
    for i,j in itertools.product(range(n), range(n)):
        for k,l in itertools.product(range(n), range(n)):
            if (i,j) != (n-1, n-1) and (k, l) != (n-1, n-1):
                bm1 = sln_basis_matrix(i, j, n, **kwargs)
                bm2 = sln_basis_matrix(k, l, n, **kwargs)
                form_val = np.trace(bm1 @ bm2)
                form[i * n + j, k * n + l] = form_val
    return form

def o_to_pgl(A, bilinear_form=np.diag([-1, 1, 1]),
             exact=True):
    r"""Return the image of an element of \(\mathrm{O}(2, 1)\) under the
    representation \(\mathrm{O}(2,1) \to \mathrm{GL}(2)\).

    On \(\mathrm{SO}(2, 1)\), this restricts to an inverse of the
    isomorphism \(\mathrm{SL}(2, \mathbb{R}) \to \mathrm{SO}(2, 1)\)
    given by the function `sl2_to_so21`.

    Parameters
    ----------
    A : ndarray
        Array of shape `(..., 3, 3)` giving a matrix (or array of
        matrices) preserving a bilinear form of signature (2, 1).
    bilinear_form : ndarray
        3x3 matrix giving the bilinear form preserved by `A`. By
        default, the diagonal form `diag(-1, 1, 1)`.

    Returns
    -------
    result : ndarray
        Array of shape `(..., 2, 2)` giving the image of `A` under
        this representation.

    """
    conj = utils.identity(3, like=A)
    conj_i = utils.identity(3, like=A)

    if bilinear_form is not None:
        bilinear_form = utils.array_like(bilinear_form,
                                         like=A)

        two = utils.number(2, like=A)
        killing_conj = utils.array_like([[ 0, -1, -1],
                                         [-2,  0,   0],
                                         [ 0,  1, -1]],
                                        like=A)

        killing_conj = killing_conj / two

        form_conj, form_conj_i = utils.diagonalize_form(
            bilinear_form,
            order_eigenvalues="minkowski",
            reverse=True,
            with_inverse=True,
            compute_exact=exact
        )

        conj = form_conj @ utils.invert(killing_conj)
        conj_i = killing_conj @ form_conj_i

    A_d = conj_i @ A @ conj

    a = np.sqrt(np.abs(A_d[0, 0]))
    b = np.sqrt(np.abs(A_d[0, 2]))
    c = np.sqrt(np.abs(A_d[2, 0]))
    d = np.sqrt(np.abs(A_d[2, 2]))

    # TODO: make this vector-safe, right now the docstring is a lie
    if A_d[0][1] < 0:
        b = b * -1
    if A_d[1][0] < 0:
        c = c * -1
    if A_d[1][2] * A_d[0][1] < 0:
        d = d * -1

    return np.array([[a, b],
                     [c, d]])

def sl2_to_so21(A):
    r"""Return the image of an element of \(\mathrm{SL}(2, \mathbb{R})\)
    under the isomorphism \(\mathrm{SL}(2, \mathbb{R}) \to
    \mathrm{SO}(2,1)\).

    Here \(\mathrm{SO}(2,1)\) preserves the symmetric bilinear form
    determined by the matrix `diag(-1, 1, 1)` (in the standard basis on
    \(\mathbb{R}^3\)).

    An inverse for this representation is given by the function
    `o_to_pgl`.

    Parameters
    ----------
    A : ndarray
        Array of shape `(..., 2, 2)` giving a matrix (or array of
        matrices) in \(\mathrm{SL}(2, \mathbb{R})\).

    Returns
    -------
    result : ndarray
        Array of shape `(..., 3, 3)` giving the image of `A` under the
        representation.

    """
    killing_conj = utils.array_like([[-0, -1, -0],
                                     [-1, -0,  1],
                                     [-1, -0, -1]],
                                    like=A)

    permutation = utils.permutation_matrix((2,1,0), like=A)

    A_3 = sl2_irrep(A, 3)

    return (permutation @ killing_conj @ A_3 @
            utils.invert(killing_conj) @ permutation)
