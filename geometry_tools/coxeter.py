"""Work with Coxeter groups and their representations.
"""
import warnings
from collections import defaultdict

import numpy as np

from geometry_tools.representation import Representation
from geometry_tools.projective import GeometryError
from geometry_tools import hyperbolic
from geometry_tools import utils
from geometry_tools.automata import fsa, coxeter_automaton

GENERATOR_NAMES = "abcdefghijklmnopqrstuvwxyz"

class CoxeterGroup:
    def __init__(self, diagram=None, matrix=None):
        """Construct a Coxeter group.

        The constructor accepts data determining a Coxeter group in
        the form of either a Coxeter diagram (specified as an iterable
        of tuples) or a Coxeter matrix (a symmetric array of
        integers).

        Parameters
        ----------
        diagram: iterable of tuples
            A Coxeter diagram, given as an iterable of triples of the
            form `(generator1, generator2, order)`. Each generator can
            be any hashable object. `order` is an integer. If it is
            positive, then it represents the order of the product of
            generator1 and generator2. If it is zero or negative, the
            order of this product is infinite.

        matrix: ndarray of ints
            The Coxeter matrix defining this Coxeter system. This must
            be a symmetric matrix of integers, with 1's on the
            diagonal.

    """
        if diagram is None and matrix is None:
            raise ValueError("Must provide data to construct a Coxeter group")

        if diagram is not None and matrix is not None:
            warnings.warn("Redundant Coxeter data specified; ignoring Coxeter matrix and"
                          " constructing from diagram.")

        if diagram is not None:
            self.from_diagram(diagram)
        elif matrix is not None:
            self.from_coxeter_matrix(matrix)

        self._compute_bilinear_form()

    def _compute_bilinear_form(self):
        adjusted_cox_matrix = np.array(self.coxeter_matrix, dtype=float)
        adjusted_cox_matrix[adjusted_cox_matrix <= 0] = 0.5

        self.bilinear_form = -1 * np.cos(np.pi / adjusted_cox_matrix)

    def from_diagram(self, diagram):
        self.generators = defaultdict(dict)

        for edge in diagram:
            g1, g2, order = edge
            self.generators[g1][g2] = order
            self.generators[g2][g1] = order

        self.generator_index = {g:i for i, g in enumerate(self.generators)}
        self.ordered_gens = [None] * len(self.generators)
        for g, i in self.generator_index.items():
            self.ordered_gens[i] = g
            self.generators[g][g] = 1

        self.coxeter_matrix = np.array([
            [self.generators[g1][g2] for g2 in self.ordered_gens]
            for g1 in self.ordered_gens
        ])

    def __str__(self):
        return "{} with matrix: \n{}".format(
            self.__class__.__name__, self.coxeter_matrix
        )


    def from_coxeter_matrix(self, matrix):
        _matrix = np.array(matrix)
        if (_matrix.T - _matrix != 0).any():
            raise GeometryError("Coxeter matrix must be symmetric")

        if (_matrix.astype(int) != _matrix).any():
            raise GeometryError("Coxeter matrix must have integer entries")

        if (np.diag(_matrix) != 1).any():
            warnings.warn("Coxeter matrix should have 1's on the diagonal")

        self.coxeter_matrix = _matrix
        self.generators = defaultdict(dict)

        n = len(_matrix)
        self.generator_index = {
            GENERATOR_NAMES[i]:i for i in range(n)
        }
        self.ordered_gens = GENERATOR_NAMES[:n]

        for i, row in enumerate(_matrix):
            for j, order in enumerate(row):
                self.generators[GENERATOR_NAMES[i]][GENERATOR_NAMES[j]] = order

    def cartan_representation(self, cartan_matrix,
                              rename_generators=False,
                              diagonalize=False,
                              order_eigenvalues="signed"):
        """Find a representation of this Coxeter group determined by a Cartan
        matrix.

        Parameters
        ----------
        cartan_matrix : ndarray
            The n*n matrix determining a representation of this
            Coxeter group into PGL(n, R), generated by reflections.
        rename_generators : bool
            If True, then rename the generators 'a', 'b', 'c',
            etc. for the representation If False (the default), use
            the provided names for the generators.
        diagonalize : bool
            If True, and the specified Cartan matrix is symmetric,
            conjugate the computed representation so that it preserves
            a diagonal symmetric bilinear form with unit-modulus
            eigenvalues.
        order_eigenvalues: {'signed', 'minkowski'}
            If diagonalize is True, specify how to order the diagonal
            basis for the form. See the documentation for
            utils.diagonalize_form for details.

        Returns
        -------
        representation.Representation
            Representation determined by the Cartan matrix

        Raises
        ------
        ValueError
            Raised if rename_generators is False and any generator has
            a name which is not a single character.

        """
        num_gens = len(self.generators)
        rep = Representation()

        for i, gen in enumerate(self.ordered_gens):
            basis_vec = np.zeros(num_gens)
            basis_vec[i] = 1.0
            diagonal = np.diag(basis_vec)

            g_name = gen
            if rename_generators:
                g_name = GENERATOR_NAMES[i]

            rep[g_name] = (
                np.identity(num_gens) - diagonal @ cartan_matrix
            )

        if not diagonalize:
            return rep

        #TODO: check if Cartan matrix is symmetric for this step
        W = utils.diagonalize_form(
            cartan_matrix / 2,
            order_eigenvalues=order_eigenvalues
        )

        return rep.compose(
            lambda mat: utils.invert(W) @ mat @ W
        )

    def geometric_representation(self, **kwargs):
        """Get the geometric representation of the Coxeter group, i.e. the
        dual of the canonical representation.

        See the documentation for CoxeterGroup.cartan_representation
        for descriptions of additional keyword arguments.

        Returns
        -------
        representation.Representation
            The geometric representation for this Coxeter group, which
            acts discrete faithfully on a convex subset of projective
            space RP^{n-1}.  This representation always preserves the
            symmetric bilinear form determined by the cosine matrix
            (which is in turn determined by the Coxeter matrix).

        """
        return self.cartan_representation(2 * self.bilinear_form, **kwargs)

    def canonical_representation(self, **kwargs):
        """Get the canonical representation of this Coxeter group into PGL(n,
        R).

        See the documentation for CoxeterGroup.cartan_representation
        for descriptions of additional keyword arguments.

        Returns
        -------
        representation.Representation
            The canonical (Tits) representation for this Coxeter
            group.

        """
        return self.geometric_representation(**kwargs).compose(
            lambda mat: utils.invert(mat.T)
        )


    def cartan_matrix(self, parameters):
        """Get a Cartan matrix for this Coxeter group by specifying
        off-diagonal parameters.

        Parameters
        ----------
        parameters : dict or ndarray
            dictionary specifying free parameters for a Cartan matrix,
            i.e. entries of the Cartan matrix corresponding to pairs
            of generators s,t where the group element s*t has infinite
            order.

            Parameters are specified either as a dictionary or a
            matrix. To specify parameters as a dictionary, use the
            format

            {(i,j):v, (k, l): u, ...}

            where i,j,k,l etc. are generator indices, and v, u,
            etc. are values of the Cartan matrix.

            To specify parameters as a matrix, pass an n*n matrix with
            all entries zero except for the free parameters in the
            Cartan matrix.

            For either format, if the (i,j) entry of the Cartan matrix
            is set, but the (j,i) entry of the matrix is not set (or
            set to zero), then this function assumes that the intended
            Cartan matrix is symmetric and will set the (j,i) entry
            equal to the (i,j) entry.

        Returns
        -------
        ndarray
            Cartan matrix, specified by the parameters as described above.

        """
        cartan = 2 * self.bilinear_form[:]

        def specified(index):
            try:
                return parameters[index] != 0
            except KeyError:
                return False

        for index in zip(*np.nonzero(self.coxeter_matrix < 0)):
            if specified(index):
                cartan[index] = parameters[index]
                r_index = index[::-1]
                if not specified(r_index):
                    cartan[r_index] = parameters[index]

        return cartan

    def tits_vinberg_rep(self, parameters, **kwargs):
        """Get a representation of this Coxeter group determined by free
        parameters in a Cartan matrix.

        See the documentation for CoxeterGroup.cartan_representation
        for descriptions of additional keyword arguments.

        Parameters
        ----------
        parameters : dict or ndarray
            Free parameters in a Cartan matrix, in the format expected
            by the function Coxeter.cartan_matrix.

        Returns
        -------
        representation.Representation
            Representation determined by the Cartan matrix specified
            by the given parameters.

        """
        cartan = self.cartan_matrix(parameters)
        return self.cartan_representation(cartan, **kwargs)

    def hyperbolic_rep(self, **kwargs):
        """Get a representation of this Coxeter group in PO(d, 1).

        This function assumes that the geometric representation of the
        Coxeter group preserves a bilinear form of signature (d, 1).

        See the documentation for CoxeterGroup.cartan_representation
        for descriptions of additional keyword arguments.

        Returns
        -------
        hyperbolic.HyperbolicRepresentation
            Representation of the Coxeter group by isometries in
            d-dimensional hyperbolic space, where d+1 is the number of
            generators of the group.

        """
        matrix_rep = self.geometric_representation(
            diagonalize=True,
            order_eigenvalues="minkowski",
            **kwargs
        )

        return hyperbolic.HyperbolicRepresentation(matrix_rep)

    def automaton(self, shortlex=True, even_length=False):
        """Get a finite-state automaton accepting geodesic words in the
        Coxeter group.

        Parameters
        ----------
        shortlex : bool
            If True (the default), return an automaton which only
            accepts shortlex geodesic representatives of elements in
            the group. If False, return an automaton which only
            accepts geodesic words, and accepts at least one geodesic
            word per element.
        even_length : bool
            If True, return an automaton only accepting words of even
            length.

        Returns
        -------
        fsa.FSA
            finite-state automaton accepting geodesic words.

    """
        aut = coxeter_automaton.generate_automaton_coxeter_matrix(
            self.coxeter_matrix,
            shortlex
        )
        aut.rename_generators(self.ordered_gens)

        if even_length:
            return aut.even_automaton()

        return aut

    def standard_subgroup(self, generators):
        """Get a standard subgroup of the Coxeter group, i.e. a subgroup
        generated by a subset of the standard set.

        Parameters
        ----------
        generators : iterable
            generating set for the standard subgroup

        Returns
        -------
        CoxeterGroup
            Standard (Coxeter) subgroup with the given generating set
        """

        subdiagram = []
        for g1, g2s in self.generators.items():
            if g1 not in generators:
                continue
            for g2, order in g2s.items():
                if g2 not in generators:
                    continue
                subdiagram.append((g1, g2, order))

        return CoxeterGroup(diagram=subdiagram)

class TriangleGroup(CoxeterGroup):
    """Convenience class for building a triangle group.

    """
    def __init__(self, vertex_params):
        v1, v2, v3 = vertex_params
        CoxeterGroup.__init__(
            self, [
                ['a', 'b', v1],
                ['b', 'c', v2],
                ['c', 'a', v3]
            ]
        )
