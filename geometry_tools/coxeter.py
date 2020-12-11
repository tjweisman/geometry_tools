from collections import defaultdict

import numpy as np

from geometry_tools.representation import Representation
from geometry_tools import utils

GENERATOR_NAMES = "abcdefghijklmnopqrstuvwxyz"

class CoxeterGroup:
    def __init__(self, diagram):
        """diagram is an iterable of triples of the form (generator1,
        generator2, order). order < 0 is interpreted as infinity.

        """

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

        self.bilinear_form = -1 * np.cos(np.pi / self.coxeter_matrix)

    def canonical_representation(self):
        num_gens = len(self.generators)
        rep = Representation(GENERATOR_NAMES[:num_gens])

        for i, gen in enumerate(self.ordered_gens):
            basis_vec = np.zeros(num_gens)
            basis_vec[i] = 1.0
            diagonal = np.diag(basis_vec)
            rep[GENERATOR_NAMES[i]] = (
                np.identity(num_gens) - 2 * diagonal @ self.bilinear_form
            )

        return rep

    def diagonal_rep(self, order_eigenvalues="signed"):
        eigs, U = np.linalg.eigh(self.bilinear_form)
        D = np.diag(1 / np.sqrt(np.abs(eigs)))

        perm = np.identity(len(self.generators))

        if order_eigenvalues and order_eigenvalues == "signed":
            perm = utils.permutation_matrix(np.argsort(eigs))

        W = U @ D @ np.linalg.inv(perm)

        rep = self.canonical_representation()
        for g in self.generators:
            rep[g] = np.linalg.inv(W) @ rep[g] @ W

        return rep

class TriangleGroup(CoxeterGroup):
    def __init__(self, vertex_params):
        v1, v2, v3 = vertex_params
        CoxeterGroup.__init__(
            self, [
                ['a', 'b', v1],
                ['b', 'c', v2],
                ['c', 'a', v3]
            ]
        )
