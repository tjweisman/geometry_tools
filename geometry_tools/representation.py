"""Work with group representations into finite-dimensional vector
spaces, using numerical matrices.

"""

import re
import itertools

import numpy as np
from scipy.special import binom

from . import utils

class RepresentationException(Exception):
    pass

class Representation:
    """Model a representation for a finitely generated group
    representation into GL(n).

    Really this is just a convenient way of mapping words in the
    generators to matrices - there's no group theory being done here
    at all.

    """

    @staticmethod
    def invert_gen(generator):
        if re.match("[a-z]", generator):
            return generator.upper()
        else:
            return generator.lower()

    @property
    def dim(self):
        return self._dim

    def elements(self, max_length):
        for word in self.free_words_less_than(max_length):
            yield (word, self[word])

    def free_words_of_length(self, length):
        if length == 0:
            yield ""
        else:
            for word in self.free_words_of_length(length - 1):
                for generator in self.generators:
                    if len(word) == 0 or generator != self.invert_gen(word[-1]):
                        yield word + generator

    def free_words_less_than(self, length):
        for i in range(length):
            for word in self.free_words_of_length(i):
                yield word

    def semi_gens(self):
        for gen in self.generators:
            if re.match("[a-z]", gen):
                yield gen

    def __init__(self, generator_names=[], normalization_step=-1):
        self.generators = {name[0].lower():None
                           for name in generator_names}

        for gen in list(self.generators):
            self.generators[gen.upper()] = None

        self.normalization_step = normalization_step

        self._dim = None

    def normalize(self, matrix):
        """function to force a matrices into a subgroup of GL(d,R)
        """
        return matrix

    def _word_value(self, word):
        matrix = np.identity(self._dim)
        for i, letter in enumerate(word):
            matrix = matrix @ self.generators[letter]
            if (self.normalization_step > 0 and
                (i % self.normalization_step) == 0):
                matrix = self.normalize(matrix)
        return matrix

    def __getitem__(self, word):
        return self._word_value(word)

    def __setitem__(self, generator, matrix):
        shape = matrix.shape

        if self._dim is None:
            self._dim = shape[0]
        if shape[0] != shape[1]:
            raise RepresentationException("use square matrices")
        if shape[0] != self._dim:
            raise RepresentationException("use matrices of matching dimensions")

        self.generators[generator] = matrix
        self.generators[self.invert_gen(generator)] = np.linalg.inv(matrix)

    def tensor_product(self, rep):
        if set(rep.generators) != set(self.generators):
            raise RepresentationException(
                "Cannot take a tensor product of a representation of groups with "
                "different presentations"
            )
        else:
            product_rep = Representation()
            for gen in self.semi_gens():
                tens = np.tensordot(self[gen], rep[gen], axes=0)
                elt = np.concatenate(np.concatenate(tens, axis=1), axis=1)
                product_rep[gen] = np.matrix(elt)
            return product_rep

    def symmetric_square(self):
        tensor_rep = self.tensor_product(self)
        incl = symmetric_inclusion(self._dim)
        proj = symmetric_projection(self._dim)
        square_rep = Representation()
        for g in self.semi_gens():
            square_rep[g] = proj * tensor_rep[g] * incl

        return square_rep



def sym_index(i,j, n):
    if i > j:
        i,j = j,i
    return int((n - i) * (n - i  - 1) / 2 + (j - i))

def tensor_pos(i, n):
    return int(i / n), i % n

def tensor_index(i,j,n):
    return i * n + j

def symmetric_inclusion(n):
    incl_matrix = np.zeros((n * n, int(n * (n + 1) / 2)))
    for i in range(n):
        for j in range(n):
            si = sym_index(i, j, n)
            ti = tensor_index(i, j, n)
            incl_matrix[ti][si] = 1/2 + (i == j) * 1/2

    return np.matrix(incl_matrix)

def symmetric_projection(n):
    proj_matrix = np.zeros((int(n * (n + 1) / 2), n * n))
    for i in range(n * n):
        u, v = tensor_pos(i,n)
        proj_matrix[sym_index(u, v, n)][i] = 1

    return np.matrix(proj_matrix)

def o_to_pgl(A, bilinear_form=np.diag((-1, 1, 1))):
    """the isomorphism SO(2,1) --> PSL(2), assuming the matrix A is a 3x3
    matrix determining a linear map in a basis where the symmetric
    bilinear form has matrix diag(-1, 1, 1).

    """
    conj = np.eye(3)
    conj_i = np.eye(3)

    if bilinear_form is not None:
        killing_conj = np.array([[ 0. , -0.5, -0.5],
                                 [-1. ,  0. ,  0. ],
                                 [ 0. ,  0.5, -0.5]])

        form_conj = utils.diagonalize_form(bilinear_form,
                                      order_eigenvalues="minkowski",
                                      reverse=True)

        conj = form_conj @ np.linalg.inv(killing_conj)
        conj_i = killing_conj @ np.linalg.inv(form_conj)

    A_d = conj_i @ A @ conj

    a = np.sqrt(np.abs(A_d[0][0]))
    b = np.sqrt(np.abs(A_d[2][0]))
    c = np.sqrt(np.abs(A_d[0][2]))
    d = np.sqrt(np.abs(A_d[2][2]))

    if A_d[0][1] < 0:
        b = b * -1
    if A_d[1][0] < 0:
        c = c * -1
    if A_d[1][2] * A_d[0][1] < 0:
        d = d * -1

    return np.array([[a, b],
                     [c, d]])

def sl2_to_so21(A):
    """the isomorphism SL(2,R) to SO(2,1) via the adjoint action, where
    SO(2,1) preserves the symmetric bilinear form with matrix diag(-1, 1,
    1)"""
    killing_conj = np.array([[-0., -1., -0.],
                             [-1., -0.,  1.],
                             [-1., -0., -1.]])
    permutation = utils.permutation_matrix((2,1,0))

    A_3 = psl_irrep(A, 3)
    return (permutation @ killing_conj @ A_3 @
            np.linalg.inv(killing_conj) @ permutation)

def psl_irrep(A, dim):
    """the irreducible representation from SL(2) to SL(dim) (via action on
    homogeneous polynomials)

    """

    a = A[..., 0, 0]
    b = A[..., 0, 1]
    c = A[..., 1, 0]
    d = A[..., 1, 1]

    im = np.zeros(A.shape[:-2] +(dim, dim))
    n = dim - 1
    for k in range(dim):
        for j in range(dim):
            for i in range(max(0, j - n + k), min(j+1, k+1)):
                im[..., j,k] += (binom(k,i) * binom(n - k, j - i)
                          * a**i * c**(k - i) * b**(j - i)
                          * d**(n - k - j + i))
    return im
