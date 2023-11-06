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

def o_to_pgl(A, bilinear_form=np.diag([-1, 1, 1])):
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
            with_inverse=True
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

def block_include(A, dimension):
    A_dim = A.shape[-1]
    arr = utils.zeros(A.shape[:-2] + (dimension, dimension),
                      like=A)

    arr[..., :A_dim, :A_dim] = A
    arr[..., A_dim:, A_dim:] = utils.identity(dimension - A_dim, like=A)
    return arr

def bilinear_form_differential(form, like=None, **kwargs):
    if like is None:
        like = form
    diff = linear_matrix_action(
        lambda mat: mat.T @ form + form @ mat,
        form.shape[-1],
        like=like, **kwargs
    )
    return diff

def subspace_action(mat, subspace, broadcast="elementwise", **kwargs):
    subspace_mats = subspace

    #can't use atleast_2d since we want a column vector here
    if len(subspace.shape) == 1:
        subspace_mats = np.expand_dims(subspace, axis=-1)

    if mat.shape[-1] != mat.shape[-2]:
        raise ValueError("Matrix must be square to compute restriction to a subspace")

    if subspace_mats.shape[-2] != mat.shape[-1]:
        raise ValueError((
            "Matrix of shape {} does not act on a subspace in an ambient space of "
            "dimension {}"
        ).format(mats.shape[-1], subspace_mats.shape[-2])
        )

    subspace_dim = subspace_mats.shape[-1]

    img_subspace = utils.matrix_product(
        mat, subspace_mats, broadcast=broadcast
    )

    img_mat = np.concatenate(
        [subspace, img_subspace],
        axis=-1
    )

    try:
        kernel = utils.kernel(img_mat, **kwargs)
        if (kernel.shape[-2] != 2 * subspace_dim or
            kernel.shape[-1] != subspace_dim):
            raise ValueError
    except ValueError:
        raise ValueError(
            "Given matri(x/ces) does not preserve the given subspace(s)"
        )

    left_coeffs = kernel[..., :subspace_dim, :]
    right_coeffs = -kernel[..., subspace_dim:, :]
    right_coeffs_inv = utils.invert(right_coeffs, **kwargs)

    return utils.matrix_product(left_coeffs, right_coeffs_inv)

def form_adjoint_action(mat, form, inv=None, **kwargs):
    lie_alg_kernel_mat = lie.bilinear_form_differential(form, **kwargs)
    lie_alg_basis = utils.kernel(lie_alg_kernel_mat, **kwargs)
    gln_adjoint_mat = lie.gln_adjoint(mat, inv=inv, **kwargs)
    return lie.subspace_action(gln_adjoint_mat,
                               lie_alg_basis,
                               **kwargs)

def slc_to_slr(mat, **kwargs):
    if mat.shape[-1] != mat.shape[-2]:
        raise ValueError("Cannot embed non-square matrices into SL(n, R)")

    dim = mat.shape[-1]

    result = utils.zeros(mat.shape[:-2] + (2 * dim, 2 * dim),
                         like=mat, **kwargs)
    result[..., :dim, :dim] = utils.real(mat)
    result[..., :dim, dim:] = -utils.imag(mat)
    result[..., dim:, :dim] = utils.imag(mat)
    result[..., dim:, dim:] = utils.real(mat)

    return result

def herm2_bilinear_form(**kwargs):
    """Return the bilinear form given by the determinant on the (real)
    vector space of 2x2 Hermitian matrices.

    The space of Hermitian matrices has real basis:
    [[1, 0], [0, 0]],

    [[0, 0], [0, 1]],

    [[0, 1], [1, 0]],

    [[0, i], [-i, i]]

    """

    I = utils.unit_imag(**kwargs)
    base_ring, dtype = utils.complex_type(**kwargs)

    # we're NOT wrapping these in np.array first
    b1 = [[1, 0], [0, 0]]
    b2 = [[0, 0], [0, 1]]
    b3 = [[0, 1], [1, 0]]
    b4 = [[0, I], [-I, 0]]

    basis = [
        utils.array_like(bi, base_ring=base_ring, dtype=dtype)
        for bi in [b1, b2, b3, b4]
    ]
    bilinear_form = utils.zeros((4,4), **kwargs)
    for i, M1 in enumerate(basis):
        for j, M2 in enumerate(basis):
            bilinear_form[i,j] = (
                utils.det(-(M1 + M2)) + utils.det(M1)  + utils.det(M2)
            ) / 2

    return bilinear_form

def sl2c_herm_action(mat, like=None, force_real=True, **kwargs):
    """Get the action of an element of SL(2, C) on the 4-dimensional real
    vector space of 2x2 Hermitian matrices.

    The space of Hermitian matrices has real basis:
    [[1, 0], [0, 0]],

    [[0, 0], [0, 1]],

    [[0, 1], [1, 0]],

    [[0, i], [-i, i]]

    """
    if like is None:
        like = mat

    I = utils.unit_imag(like=like, **kwargs)
    base_ring, dtype = utils.complex_type(like=like, **kwargs)

    #action on 4x4 complex matrices
    mat_map = linear_matrix_action(
        lambda A: mat @ A @ utils.conjugate(mat.swapaxes(-1, -2)), 2,
        base_ring=base_ring, dtype=dtype
    )

    basischange = utils.array_like(
        [[1, 0, 0, 0],
         [0, 0, 1, I],
         [0, 0, 1, -I],
         [0, 1, 0, 0]],
        base_ring=base_ring,
        dtype=dtype
    )

    action = utils.invert(basischange) @ mat_map @ basischange
    if force_real:
        return utils.real(action)

    return action

def sl2c_to_so31(mat, like=None, **kwargs):
    """Apply the exceptional Lie group isomorphism SL(2, C) to SO(3,1),
    where SO(3,1) preserves a diagonal form in R^4.

    """
    if like is None:
        like = mat

    basischange = utils.array_like(
        [[1, -1, 0, 0],
         [1, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]
        ], like=like, **kwargs)

    herm_action = sl2c_herm_action(mat, like=like, **kwargs)
    return utils.invert(basischange) @ herm_action @ basischange
