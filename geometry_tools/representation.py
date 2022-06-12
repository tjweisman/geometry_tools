"""Work with group representations into finite-dimensional vector
spaces, using numerical matrices.

To make a representation, instantiate the `Representation` class, and
assign numpy arrays to

    """

import re
import itertools

import numpy as np
from scipy.special import binom

from . import utils
from .automata import fsa

def semi_gens(generators):
    for gen in generators:
        if re.match("[a-z]", gen):
            yield gen

class RepresentationException(Exception):
    pass

class Representation:
    """Model a representation for a finitely generated group
    representation into GL(n).
    """

    @property
    def dim(self):
        return self._dim

    def freely_reduced_elements(self, length, maxlen=True,
                                with_words=False):
        automaton = fsa.free_automaton(list(self.semi_gens()))
        return self.automaton_accepted(automaton, length,
                                       maxlen=maxlen,
                                       with_words=with_words)

    def free_words_of_length(self, length):
        if length == 0:
            yield ""
        else:
            for word in self.free_words_of_length(length - 1):
                for generator in self.generators:
                    if len(word) == 0 or generator != utils.invert_gen(word[-1]):
                        yield word + generator

    def free_words_less_than(self, length):
        for i in range(length):
            for word in self.free_words_of_length(i):
                yield word

    def automaton_accepted(self, automaton, length,
                           end_state=None, maxlen=True,
                           precomputed=None, with_words=False):

        if precomputed is None:
            precomputed = {}

        if (length, end_state) in precomputed:
            return precomputed[(length, end_state)]

        empty_arr = np.array([]).reshape((0, self.dim, self.dim))

        if length == 0:
            if end_state is None or end_state in automaton.start_vertices:
                id_array = np.array([np.identity(self.dim)])
                if with_words:
                    return (id_array, [""])
                return id_array

            if with_words:
                return (empty_arr, [])
            return empty_arr

        if end_state is not None:
            prev_states = automaton.in_dict[end_state]

            if len(prev_states) == 0:
                if with_words:
                    return (empty_arr, [])
                return empty_arr


            matrix_list = []
            accepted_words = []
            for prev_state, labels in prev_states.items():
                for label in labels:
                    result = self.automaton_accepted(
                        automaton, length - 1,
                        end_state=prev_state,
                        maxlen=maxlen,
                        precomputed=precomputed,
                        with_words=with_words
                    )
                    if with_words:
                        matrices, words = result
                        words = [word + label for word in words]
                        accepted_words += words
                    else:
                        matrices = result

                    matrices = matrices @ self[label]
                    matrix_list.append(matrices)

            accepted_matrices = np.concatenate(matrix_list)
            if maxlen and length > 1:
                additional_result = self.automaton_accepted(
                    automaton, 1,
                    end_state=end_state,
                    maxlen=False,
                    with_words=with_words,
                    precomputed=precomputed
                )
                if with_words:
                    additional_mats, additional_words = additional_result
                    accepted_words = additional_words + accepted_words
                else:
                    additional_mats = result

                accepted_matrices = np.concatenate(
                    [additional_mats, accepted_matrices]
                )

            if with_words:
                accepted = (accepted_matrices, accepted_words)
            else:
                accepted = accepted_matrices

            precomputed[(length, end_state)] = accepted
            return accepted

        results = [
            self.automaton_accepted(
                automaton, length, end_state=vertex,
                maxlen=maxlen,
                with_words=with_words)
            for vertex in automaton.vertices()
        ]

        if with_words:
            matrix_list, word_list = zip(*results)
            accepted_words = list(itertools.chain(*word_list))
        else:
            matrix_list = results

        accepted_matrices = np.concatenate(matrix_list)
        if maxlen and length > 0:
            accepted_matrices = np.concatenate([
                [np.identity(self.dim)],
                accepted_matrices
            ])
            if with_words:
                accepted_words = [""] + accepted_words

        if with_words:
            accepted = (accepted_matrices, accepted_words)
        else:
            accepted = accepted_matrices

        return accepted

    def semi_gens(self):
        return semi_gens(self.generators.keys())

    def __init__(self, representation=None,
                 generator_names=None, normalization_step=-1):

        self._dim = None

        if representation is not None:
            if generator_names is None:
                generator_names = list(representation.generators)

            self.generators = {}
            for gen in semi_gens(generator_names):
                self[gen] = representation[gen]

            self._dim = representation._dim

        else:
            if generator_names is None:
                generator_names = []

            self.generators = {name[0].lower():None
                               for name in semi_gens(generator_names)}

            for gen in list(self.generators):
                self.generators[gen.upper()] = None

        self.normalization_step = normalization_step

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
        self.generators[utils.invert_gen(generator)] = np.linalg.inv(matrix)

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

def sl2_to_so21(A):
    """the isomorphism SL(2,R) to SO(2,1) via the adjoint action, where
    SO(2,1) preserves the symmetric bilinear form with matrix diag(-1, 1,
    1)"""
    killing_conj = np.array([[-0., -1., -0.],
                             [-1., -0.,  1.],
                             [-1., -0., -1.]])
    permutation = permutation_matrix((2,1,0))

    A_3 = psl_irrep(A, 3)
    return (permutation @ killing_conj @ A_3 @
            np.linalg.inv(killing_conj) @ permutation)

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
