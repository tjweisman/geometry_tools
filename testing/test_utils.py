import pytest
import numpy as np

from geometry_tools import utils

@pytest.fixture
def linear_map():
    return np.array([
        [3.0, 4.0, -6.0, 2.0],
        [1.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, -1.0, 0.0]
    ])

def test_kernel(linear_map):
    kernel = utils.kernel(linear_map)
    product = linear_map @ kernel
    assert np.allclose(
        np.zeros_like(product),
        product
    )

def test_symmetrize():
    matrix = np.random.rand(5,5)
    symm = utils.symmetric_part(matrix)
    asym = utils.antisymmetric_part(matrix)

    assert np.allclose(symm, symm.T)
    assert np.allclose(asym, -asym.T)
    assert np.allclose(symm + asym, matrix)
