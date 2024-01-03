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

@pytest.fixture
def rng():
    return np.random.default_rng()

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

def test_sphere_through(rng):
    num = 10
    true_center = rng.random((num, 3))
    true_radius = rng.random((num)) * 5

    directions = rng.random((num, 4, 3)) * 2 - 1
    normalized = directions / np.linalg.norm(directions, axis=-1)[..., np.newaxis]
    pts = np.expand_dims(
        true_center, axis=-2
    ) + normalized * np.expand_dims(true_radius, axis=(-1, -2))

    center, radius = utils.sphere_through(pts)
    assert np.allclose(center, true_center)
    assert np.allclose(radius, true_radius)

def test_circle_through(rng):
    true_center = rng.random((2))

    true_radius = rng.random() * 5

    directions = rng.random((3, 2)) * 2 - 1
    normalized = directions / np.linalg.norm(directions, axis=-1)[..., np.newaxis]
    pts = true_center + normalized * true_radius

    p1, p2, p3 = pts
    center, radius = utils.circle_through(p1, p2, p3)
    assert np.allclose(center, true_center)
    assert np.allclose(radius, true_radius)
