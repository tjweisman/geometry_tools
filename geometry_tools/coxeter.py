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
            be any hashable object (internally these will be mapped to
            characters a-z). `order` is an integer. If it is positive,
            then it represents the order of the product of generator1
            and generator2. If it is zero or negative, the order of
            this product is infinite.

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

    def canonical_representation(self):
        """Get the canonical representation of the Coxeter group.

        Returns
        -------
        representation.Representation
            The canonical (Tits) representation for this Coxeter
            group. This representation always preserves the symmetric
            bilinear form determined by the cosine matrix (which is in
            turn determined by the Coxeter matrix).

            Generators are mapped to 'a', 'b', 'c', etc.

        """
        num_gens = len(self.generators)
        rep = Representation()

        for i, gen in enumerate(self.ordered_gens):
            basis_vec = np.zeros(num_gens)
            basis_vec[i] = 1.0
            diagonal = np.diag(basis_vec)
            rep[GENERATOR_NAMES[i]] = (
                np.identity(num_gens) - 2 * diagonal @ self.bilinear_form
            )

        return rep

    #def tits_vinberg_rep(self, parameters):
        #TODO: actually compute a Tits-Vinberg representation based on
        #some parameters
    #    return self.canonical_representation()


    def diagonal_rep(self, order_eigenvalues="signed"):
        """Get a representation of the Coxeter group which preserves a
        diagonal symmetric bilinear form.

        This function finds a conjugate of the representation returned
        by canonical_representation so that in the standard basis on
        R^n, the representation preserves the bilinear form given by a
        diagonal matrix.

        Parameters
        ----------
        order_eigenvalues: {'signed', 'minkowski'}
            How to order the diagonal basis for the form. See the
            documentation for utils.diagonalize_form for details.

        Returns
        -------
        representation.Representation
            Diagonal representation for this Coxeter group. Generators
            are mapped to 'a', 'b', 'c', etc.

        """
        W = utils.diagonalize_form(self.bilinear_form,
                                   order_eigenvalues=order_eigenvalues)

        rep = self.canonical_representation()
        for g in self.generators:
            rep[g] = np.linalg.inv(W) @ rep[g] @ W

        return rep

    def hyperbolic_rep(self):
        """Get a representation of this Coxeter group in PO(d, 1).

        This function assumes that the canonical representation of the
        Coxeter group preserves a bilinear form of signature (d, 1).

        Returns
        -------
        hyperbolic.HyperbolicRepresentation
            Representation of the Coxeter group by isometries in
            d-dimensional hyperbolic space, where d+1 is the number of
            generators of the group. Generators are mapped to 'a',
            'b', 'c', etc.

        """
        matrix_rep = self.diagonal_rep(order_eigenvalues="signed")
        return hyperbolic.HyperbolicRepresentation(matrix_rep)

    def automaton(self, shortlex=True):
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
        return aut


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
