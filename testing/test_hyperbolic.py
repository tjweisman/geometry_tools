import pytest
import numpy as np

from geometry_tools import hyperbolic
from geometry_tools import GeometryError
from geometry_tools.hyperbolic import Model

@pytest.fixture
def points():
    return hyperbolic.Point(np.array([[1.0, 0.0, 0.2],
                                      [1.0, 0.5, -0.1],
                                      [1.0, -0.3, -0.4]]),
                            model=Model.PROJECTIVE)

@pytest.fixture(params=[points])
def hyperbolic_object(request):
    return request.param

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
