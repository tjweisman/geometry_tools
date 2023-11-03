import pytest

from geometry_tools import hyperbolic, projective, drawtools
from geometry_tools.hyperbolic import Model
from geometry_tools import GeometryError


@pytest.fixture(params=[Model.POINCARE, Model.HALFSPACE, Model.KLEIN])
def models_figure(request):
    return drawtools.HyperbolicDrawing(model=request.param)

@pytest.fixture
def h_figure():
    return drawtools.HyperbolicDrawing()

@pytest.fixture
def p_figure():
    return drawtools.ProjectiveDrawing()

def test_wrong_dimension_hyp(h_figure):
    point_1dim = hyperbolic.Point.get_origin(1)
    point_3dim = hyperbolic.Point.get_origin(3)

    with pytest.raises(GeometryError):
        h_figure.draw_point(point_1dim)

    with pytest.raises(GeometryError):
        h_figure.draw_point(point_3dim)

def test_wrong_dimension_proj(p_figure):
    point_1dim = projective.Point([0.], chart_index=0)
    point_3dim = projective.Point([0., 0., 0.], chart_index=0)

    with pytest.raises(GeometryError):
        p_figure.draw_point(point_1dim)

    with pytest.raises(GeometryError):
        p_figure.draw_point(point_3dim)
