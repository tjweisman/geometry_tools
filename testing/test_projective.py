import pytest
import numpy as np

from geometry_tools import projective
from geometry_tools import GeometryError

from common import *

@pytest.fixture
def points():
    return projective.Point(np.array([[1.0, 1.0, 1.5],
                                      [2.0, 0.2, -0.3],
                                      [1.0, 0.0, 7.5]]))
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
def polygons(polygon):
    return projective.Polygon([polygon, polygon])

@pytest.fixture
def point_array(points):
    return [projective.Point(pt) for pt in points]

@pytest.fixture
def identity():
    return projective.identity(2)

@pytest.fixture
def transform():
    return projective.Transformation(
        np.array([
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 3.5]
        ])
    )

@pytest.fixture
def transform_array(transform, identity):
    return projective.Transformation(
        [transform,
         transform @ transform,
         transform.inv(),
         identity]
    )

@pytest.fixture
def representation(transform):
    rep = projective.ProjectiveRepresentation()
    rep["a"] = transform
    return rep

@pytest.fixture
def rep_elts(representation):
    return representation.elements(["a", "aa", "A", ""])

@pytest.fixture
def point():
    return projective.Point([1.0, 0., 0.5])

def test_create_point_from_array(point):
    assert np.allclose(point.proj_data, np.array([1., 0., 0.5]))

def test_affine_coordinates():
    point = projective.Point([0.5, 0.8], chart_index=0)
    assert np.allclose(
        point.affine_coords(chart_index=0),
        np.array([0.5, 0.8])
    )

def test_update_affine_coords(points, other_points):
    other_aff = other_points.affine_coords(chart_index=1)
    points.affine_coords(other_aff, chart_index=1)
    assert np.allclose(
        points.affine_coords(chart_index=0),
        other_points.affine_coords(chart_index=0)
    )

def test_change_affine_chart(point):
    coords = point.affine_coords(chart_index=2)
    assert np.allclose(
        coords,
        np.array([2., 0.])
    )

def test_change_affine_chart_points(points):
    coords = points.affine_coords(chart_index=2)
    assert np.allclose(
        coords,
        np.array([[ 0.66666667,  0.66666667],
                  [-6.66666667, -0.66666667],
                  [ 0.13333333,  0.        ]])
    )

def test_invalid_affine_chart_point(point):
    with pytest.raises(GeometryError):
        point.affine_coords(chart_index=1)

def test_invalid_affine_chart_points(points):
    with pytest.raises(GeometryError):
        points.affine_coords(chart_index=1)

def test_point_shape(points):
    assert points.shape == (3,)
    assert points.proj_data.dtype == np.dtype('float64')

def test_copy_point(points):
    new_points = projective.Point(points)

    assert_numpy_equivalent(new_points, points)
    assert new_points.proj_data.dtype == np.dtype('float64')

def test_reconstruct_point(points):
    new_points = projective.Point(points.proj_data)

    assert_numpy_equivalent(points, new_points)
    assert new_points.proj_data.dtype == np.dtype('float64')

def test_point_from_points_array(points):
    array = [points, points, points, points]
    pt_array = projective.Point(array)
    assert pt_array.shape == (4, 3)
    assert pt_array.proj_data.dtype == np.dtype('float64')

def test_point_pair_shape(point_pair):
    assert point_pair.shape == (3,)

def test_copy_point_pair(point_pair):
    new_pair = projective.PointPair(point_pair)

    assert_numpy_equivalent(point_pair, new_pair)
    assert new_pair.proj_data.dtype == np.dtype('float64')


def test_reconstruct_point_pair(point_pair):
    new_pair = projective.PointPair(point_pair.proj_data)

    assert_numpy_equivalent(point_pair, new_pair)
    assert new_pair.proj_data.dtype == np.dtype('float64')

def test_endpoints(point_pair, points, other_points):
    end1, end2 = point_pair.get_end_pair()
    from_pair = projective.Point([end1, end2])
    from_objs = projective.Point([points, other_points])

    assert_numpy_equivalent(from_pair, from_objs)
    assert from_pair.proj_data.dtype == np.dtype('float64')

def test_polygon(polygon, points):
    assert polygon.shape == ()

    assert_numpy_equivalent(
        polygon.get_vertices(),
        points
    )

    assert polygon.proj_data.dtype == np.dtype('float64')

def test_polygon_from_array(polygon, point_array):
    new_poly = projective.Polygon(point_array)

    assert_numpy_equivalent(
        new_poly,
        polygon
    )
    assert new_poly.proj_data.dtype == np.dtype('float64')

def test_polygon_edges(polygon):
    assert np.allclose(
        polygon.get_edges().proj_data,
        np.array([[[ 1. ,  1. ,  1.5],
                   [ 2. ,  0.2, -0.3]],

                  [[ 2. ,  0.2, -0.3],
                   [ 1. ,  0. ,  7.5]],

                  [[ 1. ,  0. ,  7.5],
                   [ 1. ,  1. ,  1.5]]])
    )
    assert polygon.get_edges().proj_data.dtype == np.dtype('float64')

def test_compose(transform):
    assert np.allclose(
        (transform @ transform).proj_data,
        np.array([[ 1.  ,  2.  ,  0.  ],
                  [ 0.  ,  1.  ,  0.  ],
                  [ 0.  ,  0.  , 12.25]])
    )
    assert transform.proj_data.dtype == np.dtype('float64')

def test_inverse(transform):
    ident = transform @ transform.inv()
    assert np.allclose(
        ident.proj_data,
        np.identity(3)
    )
    assert ident.proj_data.dtype == np.dtype('float64')

def test_apply_to_array(transform, points):
    transformed = transform @ points
    assert transformed.shape == points.shape
    assert np.allclose(
        transformed.proj_data,
        np.array([[ 1.  ,  2.  ,  5.25],
                  [ 2.  ,  2.2 , -1.05],
                  [ 1.  ,  1.  , 26.25]])
    )
    assert transformed.proj_data.dtype == np.dtype('float64')

def test_apply_fail(transform_array, points):
    with pytest.raises(ValueError):
        transform_array @ points

def test_apply_pairwise(transform_array, points):
    transformed = transform_array.apply(points, broadcast="pairwise")
    assert transformed.shape == (4, 3)
    assert np.allclose(
        transformed.proj_data,
        np.array([[[ 1.00000000e+00,  2.00000000e+00,  5.25000000e+00],
                   [ 2.00000000e+00,  2.20000000e+00, -1.05000000e+00],
                   [ 1.00000000e+00,  1.00000000e+00,  2.62500000e+01]],

                  [[ 1.00000000e+00,  3.00000000e+00,  1.83750000e+01],
                   [ 2.00000000e+00,  4.20000000e+00, -3.67500000e+00],
                   [ 1.00000000e+00,  2.00000000e+00,  9.18750000e+01]],

                  [[ 1.00000000e+00,  0.00000000e+00,  4.28571429e-01],
                   [ 2.00000000e+00, -1.80000000e+00, -8.57142857e-02],
                   [ 1.00000000e+00, -1.00000000e+00,  2.14285714e+00]],

                  [[ 1.00000000e+00,  1.00000000e+00,  1.50000000e+00],
                   [ 2.00000000e+00,  2.00000000e-01, -3.00000000e-01],
                   [ 1.00000000e+00,  0.00000000e+00,  7.50000000e+00]]])
    )
    assert transformed.proj_data.dtype == np.dtype('float64')

def test_apply_polygon(transform, polygon):
    t_poly = transform @ polygon
    assert t_poly.shape == ()
    assert t_poly.proj_data.dtype == np.dtype('float64')

def test_apply_pairwise_polygon(transform_array, polygons):
    t_polys = transform_array.apply(polygons, broadcast="pairwise")
    assert t_polys.shape == (len(transform_array), len(polygons))
    assert t_polys.proj_data.dtype == np.dtype('float64')

def test_projective_representation(representation, transform):
    assert representation["a"].shape == ()
    assert_numpy_equivalent(
        representation["a"],
        transform
    )
    assert representation["a"].proj_data.dtype == np.dtype('float64')

def test_projective_rep_elts(rep_elts, transform_array):
    assert rep_elts.shape == (4,)
    assert_numpy_equivalent(
        rep_elts,
        transform_array
    )
    assert rep_elts.proj_data.dtype == np.dtype('float64')

def test_iterate_points(points, point_array):
    for p1, p2 in zip(points, point_array):
        assert_numpy_equivalent(p1, p2)

def test_iterate_polygons(polygons):
    for polygon in polygons:
        assert_numpy_equivalent(polygon, polygon)
