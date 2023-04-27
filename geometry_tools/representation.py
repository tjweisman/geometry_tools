"""Work with group representations into finite-dimensional vector
spaces, using numerical matrices.


To make a representation, instantiate the `Representation` class, and
assign numpy arrays to group generators (which are lowercase letters
a-z). Use square brackets to access the image of a word in the
generators.


```python
import numpy as np

from geometry_tools import representation
from geometry_tools.automata import fsa

rep = representation.Representation()
rep["a"] = np.array([
    [3.0, 0.0],
    [0.0, 1/3.0]
])

rep["aaa"]
```




    array([[27.        ,  0.        ],
           [ 0.        ,  0.03703704]])



Generator inverses are automatically assigned to capital letters:


```python
rep = representation.Representation()
rep["a"] = np.array([
    [3.0, 0.0],
    [0.0, 1/3.0]
])

rep["aA"]
```




    array([[1., 0.],
           [0., 1.]])



A common use-case for this class is to get a list of matrices
representing all elements in the group, up to a bounded word
length. The fastest way to do this is to use the built-in
`Representation.freely_reduced_elements` method, which returns a numpy
array containing one matrix for each freely reduced word in the group
(up to a specified word length).

The array of matrices is *not* typically ordered lexicographically. To
get a list of words corresponding to the matrices returned, pass the
`with_words` flag when calling
`Representation.freely_reduced_elements` (see the documentation for
that function for details).


```python
rep = representation.Representation()
rep["a"] = np.array([
    [3.0, 0.0],
    [0.0, 1/3.0]
])
rep["b"] = np.array([
    [1.0, -1.0],
    [1.0, 1.0]
]) / np.sqrt(2)


rep.freely_reduced_elements(6)
```




    array([[[ 1.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00,  1.00000000e+00]],

           [[ 7.07106781e-01,  7.07106781e-01],
            [-7.07106781e-01,  7.07106781e-01]],

           [[ 2.12132034e+00,  2.12132034e+00],
            [-2.35702260e-01,  2.35702260e-01]],

           ...,

           [[-2.12132034e+00, -2.12132034e+00],
            [ 2.35702260e-01, -2.35702260e-01]],

           [[-2.35702260e-01, -2.35702260e-01],
            [ 2.12132034e+00, -2.12132034e+00]],

           [[-1.07699575e-16, -1.00000000e+00],
            [ 1.00000000e+00,  7.51818954e-17]]])



You can speed up this process even more if you have access to a finite
state automaton which provides a unique word for each element in your
group.

For instance, to find the image of a Cayley ball of radius 10 under
the canonical representation of a (3,3,4) triangle group, you can use
the `Representation.automaton_accepted` method as follows:


```python
from geometry_tools import coxeter
from geometry_tools.automata import fsa

# create the representation and load the built-in automaton
rep = coxeter.TriangleGroup((3,3,4)).canonical_representation()
automaton = fsa.load_builtin('cox334.wa')

rep.automaton_accepted(automaton, 10)
```




    array([[[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]],

           [[ 4.44089210e-16, -1.00000000e+00,  2.41421356e+00],
            [ 1.00000000e+00, -1.00000000e+00,  1.00000000e+00],
            [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]],

           [[-1.00000000e+00,  1.00000000e+00,  1.41421356e+00],
            [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]],

           ...,

           [[-2.47279221e+01,  2.57279221e+01,  2.23137085e+01],
            [-2.91421356e+01,  2.91421356e+01,  2.71421356e+01],
            [-1.50710678e+01,  1.50710678e+01,  1.40710678e+01]],

           [[-7.24264069e+00,  7.24264069e+00,  6.82842712e+00],
            [-1.06568542e+01,  1.16568542e+01,  9.24264069e+00],
            [-5.82842712e+00,  6.82842712e+00,  4.82842712e+00]],

           [[-2.47279221e+01,  2.57279221e+01,  2.23137085e+01],
            [-1.06568542e+01,  1.16568542e+01,  9.24264069e+00],
            [-3.05563492e+01,  3.29705627e+01,  2.67279221e+01]]])

    """

import re
import itertools

import numpy as np
from scipy.special import binom

from . import utils
from .automata import fsa

def semi_gens(generators):
    """Get an iterable of semigroup generators from an iterable of group generators.

    Given a sequence of lowercase/uppercase letters, return only the
    lowercase ones.

    Parameters
    ----------
    generators : iterable of strings
        Sequence of semigroup generators

    Yields
    ------
    gen : string
        the lowercase characters in `generators`

    """
    for gen in generators:
        if re.match("[a-z]", gen):
            yield gen

class Representation:
    """Model a representation for a finitely generated group
    representation into GL(n).
    """

    @property
    def dim(self):
        return self._dim

    def freely_reduced_elements(self, length, maxlen=True,
                                with_words=False):
        """Return group elements, one for each freely reduced word
           in the generators.

        Parameters
        ----------
        length : int
            maximum length of a word to find a representation of
        maxlen : bool
            If `True` (the default), the returned array has one
            element for each word up to length `length`
            (inclusive). If `False`, only compute the image of words
            of length exactly equal to `length`.
        with_words : bool
            If `True`, also return the list of words corresponding to
            the computed matrices.

        Returns
        -------
        result : ndarray or tuple
            If `with_words` is `True`, `result` is a tuple `(elements, words)`,
            where `elements` is an ndarray of shape `(k, n, n)`,
            containing one matrix for each of the `k` freely reduced
            words in the generators, and `words` is a python list of
            strings, containing those words. If `with_words` is
            `False`, just return `elements`.

        """
        automaton = fsa.free_automaton(list(self.semi_gens()))
        return self.automaton_accepted(automaton, length,
                                       maxlen=maxlen,
                                       with_words=with_words)

    def free_words_of_length(self, length):
        """Yield freely reduced words in the generators, of a specified length.

        Parameters
        ----------
        length : int
            the length of words to return

        Yields
        ------
        word : string
            All freely reduced words in the generators of length
            `length`.

        """

        if length == 0:
            yield ""
        else:
            for word in self.free_words_of_length(length - 1):
                for generator in self.generators:
                    if len(word) == 0 or generator != utils.invert_gen(word[-1]):
                        yield word + generator

    def free_words_less_than(self, length):
        """Yield freely reduced words in the generators, up to a specified
        length.

        Parameters
        ----------
        length : int
            the maximum length of words to return

        Yields
        ------
        word : string
            All freely reduced words in the generators, up to length
            `length` (inclusive)

        """
        for i in range(length):
            for word in self.free_words_of_length(i):
                yield word

    def automaton_accepted(self, automaton, length,
                           maxlen=True, with_words=False,
                           start_state=None, end_state=None,
                           precomputed=None):
        """Return group elements representing words accepted by a
           finite-state automaton.

        Parameters
        ----------
        automaton : automata.fsa.FSA
            Finite-state automaton which determines the accepted words
        length : int
            Maximum length of a word to compute
        maxlen : bool
            if `True` (the default), compute representations of all
            accepted words of length up to `length`. If `False`, only
            compute representations of words whose length is exactly
            `length`.
        with_words : bool
            Whether to return a list of accepted words along with the
            computed array of images.
        start_state: object
            vertex of the automaton to use as the starting state for
            all accepted words. If `None`, then the default start
            vertex of the automaton is used. *Note*: at most one of
            `start_state` and `end_state` can be specified.
        end_state: object
            vertex of the automaton where all accepted paths must
            end. If `None`, then any ending state is allowed. *Note*:
            at most one of `start_state` and `end_state` can be
            specified.
        precomputed: dict
            dictionary giving precomputed values of this
            function. Keys are tuples of then form `(length, state)`,
            where `length` is an integer and `state` is a vertex of
            the automaton. If `None`, use an empty dictionary. In
            either case, the dictionary will be populated when the
            function is called.

        Returns
        -------
        result : ndarray or tuple
            If `with_words` is `True`, `result` is a tuple `(elements, words)`,
            where `elements` is an ndarray containing one
            matrix for each accepted word, and `words` is a list of
            strings containing the corresponding words. If
            `with_words` is `False`, just return `elements`.

        """

        if start_state is not None and end_state is not None:
            raise ValueError("At most one of start_state and end_state "
                             "can be specified")

        as_start = True
        state = None

        if start_state is not None:
            state = start_state
            as_start = True

        if end_state is not None:
            state = end_state
            as_start = False

        return self._automaton_accepted(automaton, length,
                                        maxlen=maxlen,
                                        with_words=with_words,
                                        state=state,
                                        as_start=as_start,
                                        precomputed=precomputed)


    def _automaton_accepted(self, automaton, length,
                           state=None, as_start=True, maxlen=True,
                           precomputed=None, with_words=False):

        if precomputed is None:
            precomputed = {}

        if (length, state) in precomputed:
            return precomputed[(length, state)]

        empty_arr = np.array([]).reshape((0, self.dim, self.dim))

        if length == 0:
            if state is None or as_start or state in automaton.start_vertices:
                id_array = np.array([np.identity(self.dim)])
                if with_words:
                    return (id_array, [""])
                return id_array

            if with_words:
                return (empty_arr, [])
            return empty_arr

        if state is None:
            as_start = True
            state = automaton.start_vertices[0]

        if as_start:
            adj_states = automaton.out_dict[state]
        else:
            adj_states = automaton.in_dict[state]

        if len(adj_states) == 0 and not as_start:
            if with_words:
                return (empty_arr, [])
            return empty_arr


        matrix_list = []
        accepted_words = []
        for adj_state, labels in adj_states.items():
            for label in labels:
                result = self._automaton_accepted(
                    automaton, length - 1,
                    state=adj_state,
                    as_start=as_start,
                    maxlen=maxlen,
                    precomputed=precomputed,
                    with_words=with_words
                )
                if with_words:
                    matrices, words = result
                    if as_start:
                        words = [label + word for word in words]
                    else:
                        words = [word + label for word in words]
                    accepted_words += words
                else:
                    matrices = result

                if as_start:
                    matrices = self._word_value(label) @ matrices
                else:
                    matrices = matrices @ self._word_value(label)
                matrix_list.append(matrices)


        accepted_matrices = np.concatenate(matrix_list)
        if maxlen:
            additional_result = self._automaton_accepted(
                automaton, 0,
                state=state,
                as_start=as_start,
                with_words=with_words
            )
            if with_words:
                additional_mats, additional_words = additional_result
                accepted_words = additional_words + accepted_words
            else:
                additional_mats = additional_result

            accepted_matrices = np.concatenate(
                [additional_mats, accepted_matrices]
            )

        if with_words:
            accepted = (accepted_matrices, accepted_words)
        else:
            accepted = accepted_matrices

        precomputed[(length, state)] = accepted
        return accepted

    def elements(self, words):
        """Get images of an iterable of words.

        Parameters
        ----------
        words : iterable of strings
            words to find the image of under the representation

        Returns
        -------
        ndarray
            numpy array of shape `(l, n, n)`, where `l` is the length
            of `words` and `n` is the dimension of the representation.

        """
        return np.array(
            [self._word_value(word) for word in words]
        )

    def semi_gens(self):
        """Iterate over group generator names for this representation.

        Only iterate over group generators (lowercase letters), not
        semigroup generators (upper and lowercase letters).

        Yields
        ------
        gen : string
            group generator names

        """
        return semi_gens(self.generators.keys())

    def __init__(self, representation=None,
                 generator_names=None):
        """
        Parameters
        ----------
        representation : Representation
            Representation to copy elements from
        generator_names : iterable of strings
            Names to use for the generators. These must be initialized
            as arrays later to use the representation properly.

        """

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

    def _word_value(self, word):
        matrix = np.identity(self._dim)
        for i, letter in enumerate(word):
            matrix = matrix @ self.generators[letter]
        return matrix

    def __getitem__(self, word):
        return self._word_value(word)

    def __setitem__(self, generator, matrix):
        shape = matrix.shape

        if self._dim is None:
            self._dim = shape[0]
        if shape[0] != shape[1]:
            raise ValueError("Matrices representing group elements must be square")
        if shape[0] != self._dim:
            raise ValueError(
                "Every matrix in the representation must have the same shape"
            )

        self.generators[generator] = matrix
        self.generators[utils.invert_gen(generator)] = np.linalg.inv(matrix)

    def compose(self, hom):
        """Get a new representation obtained by composing this representation
        with hom."""

        composed_rep = self.__class__()

        for g, image in self.generators.items():
            composed_rep[g] = hom(image)

        return composed_rep

    def tensor_product(self, rep):
        """Return a tensor product of this representation with `rep`.

        Parameters
        ----------
        rep : Representation
            Representation to tensor with.

        Raises
        ------
        ValueError
            Raised if `self` and `rep` have differing generating sets.

        Returns
        -------
        tensor: Representation
            Representation giving the tensor product of self with `rep`.

        """
        if set(rep.generators) != set(self.generators):
            raise ValueError(
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
        """Return the symmetric square of this representation.

        Returns
        -------
        square : Representation
            Symmetric square of `self`.

        """

        tensor_rep = self.tensor_product(self)
        incl = symmetric_inclusion(self._dim)
        proj = symmetric_projection(self._dim)
        square_rep = Representation()
        for g in self.semi_gens():
            square_rep[g] = proj * tensor_rep[g] * incl

        return square_rep


def sym_index(i, j, n):
    r"""Return coordinate indices for an isomorphism
    \(\mathrm{Sym}^2(\mathbb{R}^n) \to \mathbb{R}^{\binom{n}{2} + n}\).

    If \(\{e_1, \ldots, e_n\}\) is the standard basis for \(\mathbb{R}^n\),
    the isomorphism is realized by giving \(\mathrm{Sym}^2(\mathbb{R}^n)\)
    the ordered basis
    \[
        \{e_ne_n, e_{n-1}e_{n-1}, e_{n-1}e_n,
        e_{n-1}e_{n-1}, e_{n-1}e_{n-2}, e_{n-1}e_{n-3}, \ldots \}.
    \]
    Schematically this is given by the symmetric matrix

    \[\begin{pmatrix} \ddots \\
    & 3 & 4 & 5 \\
    & & 1 & 2 \\
    & & & 0
    \end{pmatrix},
    \]
    where the (i,j) entry of the matrix gives the index of basis
    element \(e_ie_j\).

    Parameters
    ----------
    i : int
        index of one of the terms in the basis monomial \(e_ie_j\) for the
        symmetric square
    j : int
        index of the other term in the basis monomial \(e_ie_j\) for the
        symmetric square
    n : int
        dimension of the underlying vector space \(\mathbb{R}^n\).

    Returns
    -------
    int
        index of the corresponding basis vector in
        \(\mathbb{R}^{\binom{n}{2} + n}\).

    """
    if i > j:
        i, j = j, i
    return int((n - i) * (n - i  - 1) / 2 + (j - i))

def tensor_pos(i, n):
    r"""Return coordinate indices for an isomorphism
    \(\mathbb{R}^{n^2} \to \mathbb{R}^n \otimes \mathbb{R}^n\).

    If \(\{e_1, \ldots, e_n\}\) is the standard basis for
    \(\mathbb{R}^n\), the isomorphism is realized by giving
    \(\mathbb{R}^n \otimes \mathbb{R}^n\) the ordered basis
    \[
    \{e_1 \otimes e_1, e_1 \otimes e_2, \ldots, e_1 \otimes e_n, e_2 \otimes e_1, \ldots, \}
    \]
    represented schematically by the matrix
    \[
        \begin{pmatrix}
            0 & 1 & \ldots \\
            n & n + 1 & \ldots\\
            \vdots
        \end{pmatrix}.
    \]
    Here the (i, j) entry of the matrix gives the index of the basis
    element \(e_i \otimes e_j\).

    The inverse of this isomorphism is given by `tensor_index`.

    Parameters
    ----------
    i : int
        index of a basis vector in \(\mathbb{R}^{n^2}\)
    n : int
        dimension of the underlying vector space \(\mathbb{R}^n\)

    Returns
    -------
    tuple
        tuple `(j, k)` determining the monomial \(e_j \otimes e_k\)
        mapped to by given the basis vector in \(\mathbb{R}^{n^2}\).

    """
    return int(i / n), i % n

def tensor_index(i,j,n):
    r"""Return coordinate indices for an isomorphism
    \(\mathbb{R}^n \otimes \mathbb{R}^n \to \mathbb{R}^{n^2}\).

    If \(\{e_1, \ldots, e_n\}\) is the standard basis for
    \(\mathbb{R}^n\), the isomorphism is realized by giving
    \(\mathbb{R}^n \otimes \mathbb{R}^n\) the ordered basis
    \[
    \{e_1 \otimes e_1, e_1 \otimes e_2, \ldots, e_1 \otimes e_n, e_2 \otimes e_1, \ldots, \}
    \]
    represented schematically by the matrix
    \[
        \begin{pmatrix}
            0 & 1 & \ldots \\
            n & n + 1 & \ldots\\
            \vdots
        \end{pmatrix}.
    \]
    Here the (i, j) entry of the matrix gives the index of the basis
    element \(e_i \otimes e_j\).

    The inverse of this isomorphism is given by `tensor_pos`.

    Parameters
    ----------
    i : int
        index of one of the terms in a basis vector \(e_i \otimes e_j\).
    j : int
        index of the other term in a basis vector \(e_i \times e_j\).
    n : int
        dimension of the underlying vector space \(\mathbb{R}^n\)

    Returns
    -------
    int
        index of a basis vector in \(\mathbb{R}^{n^2}\) mapped to by
        \(e_i \otimes e_j\).
    """
    return i * n + j

def symmetric_inclusion(n):
    r"""Return a matrix representing the linear inclusion
    \(\mathrm{Sym}^2(\mathbb{R}^n) \to \mathbb{R}^n \otimes
    \mathbb{R}^n\).

    \(\mathrm{Sym}^2(\mathbb{R}^n)\) and
    \(\mathbb{R}^n \otimes \mathbb{R}^n\)
    are respectively identified with
    \(\mathbb{R}^{\binom{n}{2} + n}\) and \(\mathbb{R}^{n^2}\) via the
    isomorphisms described in `sym_index`, `tensor_index`, and
    `tensor_pos`.

    If \(\{e_1, \ldots, e_n\}\) is the standard basis for
    \(\mathbb{R}^n\), the returned matrix gives the linear map taking
    \(e_ie_j\) to \(\frac{1}{2}(e_i \otimes e_j + e_j \otimes e_i)\),
    with respect to the bases specified above.

    Parameters
    ----------
    n : int
        Dimension of the underlying vector space \(\mathbb{R}^n\).

    Returns
    -------
    matrix : ndarray
        \(n^2 \times \binom{n}{2} + n\) array defining this linear map.

    """
    incl_matrix = np.zeros((n * n, int(n * (n + 1) / 2)))
    for i in range(n):
        for j in range(n):
            si = sym_index(i, j, n)
            ti = tensor_index(i, j, n)
            incl_matrix[ti][si] = 1/2 + (i == j) * 1/2

    return np.matrix(incl_matrix)

def symmetric_projection(n):
    r"""Return a matrix representing the linear surjection
    \(\mathbb{R}^n \otimes \mathbb{R}^n \to \mathrm{Sym}^2(\mathbb{R}^n)\).

    If \(\mathbb{R}^n\) is given the standard basis \(\{e_1, \ldots,
    e_n\}\), then this matrix represents the linear map determined by
    \(e_i \otimes e_j \mapsto e_ie_j\). The spaces
    \(\mathbb{R}^n \otimes \mathbb{R}^n\) and \(\mathrm{Sym}^2(\mathbb{R}^n)\)
    are given the ordered bases determined by the functions
    `sym_index`, `tensor_index`, and `tensor_pos`.

    Parameters
    ----------
    n : int
        Dimension of the underlying vector space \(\mathbb{R}^n\)

    Returns
    -------
    ndarray
        \(\binom{n}{2} + n \times n\) matrix representing the linear map
        in the given bases.

    """
    proj_matrix = np.zeros((int(n * (n + 1) / 2), n * n))
    for i in range(n * n):
        u, v = tensor_pos(i,n)
        proj_matrix[_sym_index(u, v, n)][i] = 1

    return np.matrix(proj_matrix)

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

    im = np.zeros(A.shape[:-2] +(n, n))
    r = n - 1
    for k in range(n):
        for j in range(n):
            for i in range(max(0, j - r + k), min(j+1, k+1)):
                im[..., j,k] += (binom(k,i) * binom(r - k, j - i)
                          * a**i * c**(k - i) * b**(j - i)
                          * d**(r - k - j + i))
    return im

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
    killing_conj = np.array([[-0., -1., -0.],
                             [-1., -0.,  1.],
                             [-1., -0., -1.]])
    permutation = utils.permutation_matrix((2,1,0))

    A_3 = sl2_irrep(A, 3)
    return (permutation @ killing_conj @ A_3 @
            np.linalg.inv(killing_conj) @ permutation)

def o_to_pgl(A, bilinear_form=np.diag((-1., 1., 1.))):
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
