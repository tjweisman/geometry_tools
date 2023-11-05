import pytest
import numpy as np

from geometry_tools import utils
from geometry_tools import lie

@pytest.fixture
def diag_form():
    return np.array([
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

@pytest.fixture
def nondiag_form(diag_form):
    conjugator = np.random.rand(3,3) * 5 - 10
    return utils.symmetric_part(
        conjugator @ diag_form @ np.linalg.inv(conjugator)
    )

@pytest.fixture
def uppertri():
    return np.array([
        [1.0, 1.0, 4.0, 1.0],
        [0.0, -2.0, -0.5, 1.6],
        [0.0, 0.0, 1.1, -6.2],
        [0.0, 0.0, 0.0, 0.2]
    ])

@pytest.fixture
def matrix_with_subspace(uppertri):
    conjugator = np.random.rand(4,4) * 5 - 10
    matrix = conjugator @ uppertri @ np.linalg.inv(conjugator)
    subspace = conjugator @ np.eye(4, 2)

    return matrix, subspace

def test_bilinear_form_alg_dim(diag_form, nondiag_form):
    diff1 = lie.bilinear_form_differential(diag_form)
    diff2 = lie.bilinear_form_differential(nondiag_form)

    assert diff1.shape == (9, 9)
    assert diff2.shape == (9, 9)

    assert min(utils.kernel(diff1).shape) == 3
    assert min(utils.kernel(diff2).shape) == 3

def test_subspace_restrict(matrix_with_subspace, uppertri):
    matrix, subspace = matrix_with_subspace
    action = lie.subspace_action(matrix, subspace)

    assert np.allclose(
        action,
        uppertri[:2, :2]
    )

def test_subspaces_restict(uppertri):
    conjugators = np.random.rand(3, 4, 4) * 5 - 10
    matrices = conjugators @ uppertri @ np.linalg.inv(conjugators)
    subspaces = conjugators @ np.eye(4, 2)

    assert subspaces.shape == (3, 4, 2)
    assert matrices.shape == (3, 4, 4)

    actions = lie.subspace_action(matrices, subspaces)

    assert actions.shape == (3, 2, 2)
    assert np.allclose(
        actions,
        uppertri[:2, :2]
    )
