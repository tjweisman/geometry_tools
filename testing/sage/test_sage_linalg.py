from sage.all import Matrix

import numpy as np
import pytest

from geometry_tools import utils

@pytest.fixture
def matrix():
    return np.array([[1, 2, 4, 6],
                     [0, 1, 1, -1],
                     [0, 2, 2, -2]], dtype="O")

@pytest.fixture
def matrices():
    return np.array([
        [[1, 2, 4, 6],
         [0, 1, 1, -1],
         [0, 2, 2, -2]],
        [[1, 2, 4, 6],
         [0, 1, 1, -1],
         [11, 3, 2, -2]]
        ], dtype="O")


def test_kernel(matrix):
    kernel = utils.kernel(matrix)
    product = matrix @ kernel

    assert kernel.shape == (4, 2)
    assert product.dtype == np.dtype("O")
    assert np.all(product == 0)
    assert isinstance(product[0,0], Integer)

def test_kernel_multiple(matrix):
    arr = np.array([matrix, matrix])
    kernel = utils.kernel(arr)
    product = utils.matrix_product(arr, kernel)

    assert kernel.shape == (2, 4, 2)
    assert product.dtype == np.dtype("O")
    assert np.all(product == 0)
    assert isinstance(product[0,0,0], Integer)


def test_non_match_dims_error(matrices):
    with pytest.raises(ValueError):
        utils.kernel(matrices)

def test_non_match_dims(matrices):
    kernels, where = utils.kernel(matrices, matching_rank=False,
                           with_loc=True)
    assert kernels[0].shape == (1, 4, 1)
    assert kernels[1].shape == (1, 4, 2)

    assert np.all(matrices[where[0]] @ kernels[0] == 0)
    assert np.all(matrices[where[1]] @ kernels[1] == 0)
