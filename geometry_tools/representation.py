import numpy as np
import re
import itertools

from scipy.special import binom

class RepresentationException(Exception):
    pass

class Representation:
    #technically, this is just a free group representation

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
        for i in range(max_length + 1):
            for word in itertools.product(self.generators, repeat=i):
                yield (''.join(word), self[word])

    def semi_gens(self):
        for gen in self.generators:
            if re.match("[a-z]", gen):
                yield gen

    def __init__(self, generator_names=[]):
        self.generators = {name[0].lower():None
                           for name in generator_names}

        for gen in list(self.generators):
            self.generators[gen.upper()] = None

        self._dim = None

    def __getitem__(self, generator):
        if len(generator) == 1:
            return self.generators[generator[0]]
        elif len(generator) > 1:
            return self[generator[0]] @ self[generator[1:]]
        else:
            return np.identity(self._dim)

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

def so_to_psl(A):
    """the isomorphism SO(2,1) --> PSL(2), assuming the matrix A is a 3x3
    matrix determining a linear map in a basis where the symmetric
    bilinear form has matrix diag(-1, -1, 1).

    """

    a = np.sqrt((A[1][1] + A[2][1] + A[1][2] + A[2][2]) / 2)
    b = (A[0][1] + A[0][2]) / (2 * a)
    c = (A[1][0] + A[2][0]) / (2 * a)
    d = (A[2][0] - A[1][0]) / (2 * b)

    return np.matrix([[a, c],
                      [b, d]])

def psl_irrep(A, dim):
    """the irreducible representation from SL(2) to SL(dim) (via action on
    homogeneous polynomials)

    """

    a = A[0,0]
    b = A[0,1]
    c = A[1,0]
    d = A[1,1]

    im = np.matrix(np.zeros((dim, dim)))
    n = dim - 1
    for k in range(dim):
        for j in range(dim):
            for i in range(max(0, j - n + k), min(j+1, k+1)):
                im[j,k] += (binom(k,i) * binom(n - k, j - i)
                          * a**i * c**(k - i) * b**(j - i)
                          * d**(n - k - j + i))
    return im
