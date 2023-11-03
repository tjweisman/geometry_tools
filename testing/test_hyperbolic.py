import pytest
import numpy as np

from geometry_tools import hyperbolic
from geometry_tools import GeometryError
from geometry_tools.hyperbolic import Model

from geometry_tools.utils.testing import *

@pytest.fixture
def points():
    return hyperbolic.Point(np.array([[1.0, 0.0, 0.2],
                                      [1.0, 0.5, -0.1],
                                      [1.0, -0.3, -0.4]]),
                            model=Model.PROJECTIVE)

@pytest.fixture
def other_points():
    return hyperbolic.Point(np.array([[1.0, -0.8, 0.05],
                                      [1.0, 0.1, -0.2],
                                      [1.0, 0.3, 0.4]]),
                            model=Model.PROJECTIVE)

@pytest.fixture
def loxodromic():
    param = 5.0
    return hyperbolic.Isometry.standard_loxodromic(2, param)

@pytest.fixture
def elliptic():
    param = np.pi / 3
    return hyperbolic.Isometry.standard_rotation(param)

@pytest.fixture
def origin_to_points(points):
    return points.origin_to()

def test_point_coordinates(points):
    assert np.allclose(
        points.coords(model=Model.KLEIN),
        np.array([[0.0, 0.2],
                  [0.5, -0.1],
                  [-0.3, -0.4]])
    )
    assert np.allclose(
        points.coords(model=Model.POINCARE),
        np.array([[ 0.        ,  0.10102051],
                  [ 0.2687836 , -0.05375672],
                  [-0.16076952, -0.21435935]])
    )
    assert np.allclose(
        points.coords(model=Model.HALFSPACE),
        np.array([[-0.2       ,  0.9797959 ],
                  [ 0.2       ,  1.72046505],
                  [ 0.30769231,  0.66617339]])
    )

def test_point_datatype(points):
    assert points.proj_data.dtype == np.dtype('float64')

def test_ideal_point_from_angle():
    pt0 = hyperbolic.IdealPoint.from_angle(0.)

    # just check that this is a lightlike point, we don't have an
    # actual standard here
    assert pt0.shape == ()
    assert np.all(hyperbolic.lightlike(pt0.proj_data))
    assert pt0.proj_data.dtype == np.dtype('float64')

    pts = hyperbolic.IdealPoint.from_angle([0., np.pi])
    assert pts.shape == (2,)
    assert np.all(hyperbolic.lightlike(pts.proj_data))
    assert pts.proj_data.dtype == np.dtype('float64')

def test_distances(points, other_points):
    assert np.allclose(
        points.distance(other_points),
        np.array([1.11585227, 0.463157  , 1.09861229])
    )

def test_unit_tv_to(points, other_points):
    unit_tvs = points.unit_tangent_towards(other_points)
    assert unit_tvs.shape == points.shape

def test_point_along(points, other_points):
    unit_tvs = points.unit_tangent_towards(other_points)
    distances = points.distance(other_points)

    along = unit_tvs.point_along(distances)

    assert along.shape == other_points.shape
    assert np.allclose(
        along.coords(model=Model.KLEIN),
        other_points.coords(model=Model.KLEIN)
    )

def test_get_origin():
    origin = hyperbolic.Point.get_origin(2, dtype=int)
    assert origin.proj_data.dtype == int
    assert np.all(origin.coords(model=Model.KLEIN) == 0)

    origin = hyperbolic.Point.get_origin(4, dtype=np.dtype('float64'))
    assert origin.proj_data.dtype == np.dtype('float64')
    assert np.all(origin.coords(model=Model.KLEIN) == 0)

def test_origin_arr():
    origin = hyperbolic.Point.get_origin(3, shape=(2,3))

    assert origin.shape == (2,3)
    assert np.all(origin.coords(model=Model.KLEIN) == 0.)

def test_origin_to_point(origin_to_points, points):
    origin = hyperbolic.Point.get_origin(2)
    assert_numpy_equivalent(
        origin_to_points @ origin,
        points
    )

def test_elliptic_fixpoints(elliptic, origin_to_points, points):
    conj_elliptic = origin_to_points @ elliptic @ origin_to_points.inv()
    fixpoints = conj_elliptic.fixed_point()

    np.allclose(
        fixpoints.coords(model=Model.KLEIN),
        points.coords(model=Model.KLEIN)
    )

def test_loxodromic_fixpoints(loxodromic, elliptic):
    iso = elliptic.inv() @ loxodromic @ elliptic
    fixpoints = iso.fixed_point_pair()

    ell_param = np.pi / 3

    expected_fixpoint_coords = np.array([
        [np.cos(ell_param), -np.sin(ell_param)],
        [-np.cos(ell_param), np.sin(ell_param)]
    ])

    assert (
        np.allclose(
            fixpoints.coords(model=Model.KLEIN),
            expected_fixpoint_coords) or
        np.allclose(
            fixpoints.coords(model=Model.KLEIN),
            expected_fixpoint_coords[::-1]
        )
    )
