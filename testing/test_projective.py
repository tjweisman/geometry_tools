import pytest
import numpy as np

from geometry_tools import projective

@pytest.fixture
def points():
    return projective.Point(np.array([[1.0, 1.0, 1.5],
                                      [2.0, 0.2, -0.3],
                                      [1.0, 0.6, 7.5]]))
@pytest.fixture
def other_points():
    return projective.Point(np.array([
        [1.0, 6.0, 0.9],
        [0.5, 0.1, -11.1],
        [3.0, 1.0, 1.5]
    ]))

@pytest.fixture
def point_pair(points, other_points):
    return projective.PointPair(points, other_points)

@pytest.fixture
def polygon(points):
    return projective.Polygon(points)

@pytest.fixture
def point_array(points):
    return [projective.Point(pt) for pt in points]

def test_create_point_from_array():
    point = projective.Point([1.0, 0., 0.5])

    assert np.allclose(point.proj_data, np.array([1., 0., 0.5]))

def test_projective_coordinates():
    point = projective.Point([0.5, 0.8], chart_index=0)
    assert np.allclose(
        point.affine_coords(chart_index=0),
        np.array([0.5, 0.8])
    )

def test_point_shape(points):
    assert points.shape == (3,)

def test_copy_point(points):
    new_points = projective.Point(points)
    assert np.allclose(new_points.proj_data,
                       points.proj_data)

def test_reconstruct_point(points):
    new_points = projective.Point(points.proj_data)
    assert np.allclose(new_points.proj_data,
                       points.proj_data)

def test_point_from_points_array(points):
    array = [points, points, points, points]
    pt_array = projective.Point(array)
    assert pt_array.shape == (4, 3)

def test_point_pair_shape(point_pair):
    assert point_pair.shape == (3,)

def test_copy_point_pair(point_pair):
    new_pair = projective.PointPair(point_pair)
    assert np.allclose(point_pair.proj_data,
                       new_pair.proj_data)

def test_reconstruct_point_pair(point_pair):
    new_pair = projective.PointPair(point_pair.proj_data)
    assert np.allclose(point_pair.proj_data,
                       new_pair.proj_data)

def test_endpoints(point_pair, points, other_points):
    end1, end2 = point_pair.get_end_pair()
    from_pair = projective.Point([end1, end2])
    from_objs = projective.Point([points, other_points])
    assert from_pair.shape == from_objs.shape
    assert np.allclose(from_pair.proj_data, from_objs.proj_data)

def test_polygon(polygon, points):
    assert polygon.shape == ()
    assert polygon.get_vertices().shape == points.shape
    assert np.allclose(
        polygon.get_vertices().proj_data,
        points.proj_data
    )

def test_polygon_from_array(polygon, point_array):
    new_poly = projective.Polygon(point_array)
    assert new_poly.shape == polygon.shape
    assert np.allclose(
        new_poly.proj_data,
        polygon.proj_data
    )

def test_polygon_edges(polygon):
    assert np.allclose(
        polygon.get_edges().proj_data,
        np.array([[[ 1. ,  1. ,  1.5],
                   [ 2. ,  0.2, -0.3]],

                  [[ 2. ,  0.2, -0.3],
                   [ 1. ,  0.6,  7.5]],

                  [[ 1. ,  0.6,  7.5],
                   [ 1. ,  1. ,  1.5]]])
    )
