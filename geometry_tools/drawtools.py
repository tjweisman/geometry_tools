"""geometry_tools.drawtools

Provides some useful functions for turning objects from the hyperbolic
module into matplotlib figures.

"""

import copy

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc, PathPatch, Rectangle
from matplotlib.collections import LineCollection, PolyCollection, EllipseCollection

from matplotlib.transforms import Affine2D
from matplotlib.path import Path

from geometry_tools import hyperbolic, utils
from geometry_tools.hyperbolic import Model

#I played around with this a bit, but it's an eyeball test
#TBH. Determines the radius at which we start approximating circular
#arcs with straight lines.
RADIUS_THRESHOLD = 80

#how far apart points can be before we decide that we ordered the
#polygon wrong
DISTANCE_THRESHOLD = 1e-5

DRAW_NEIGHBORHOOD = 0.1

#this is a bit unpythonic since these are meant to be constants
def default_model_limits(model):
    if model == Model.POINCARE or model == Model.KLEIN:
        return ((-1 - DRAW_NEIGHBORHOOD, 1 + DRAW_NEIGHBORHOOD),
                (-1 - DRAW_NEIGHBORHOOD, 1 + DRAW_NEIGHBORHOOD))

    if model == Model.HALFSPACE:
        return ((-8., 8.),
                (-1 * DRAW_NEIGHBORHOOD, 10.))

class DrawingError(Exception):
    """Thrown if we try and draw an object in a model which we haven't
    implemented yet.

    """
    pass

class HyperbolicDrawing:
    def __init__(self, figsize=8,
                 ax=None,
                 fig=None,
                 facecolor="aliceblue",
                 edgecolor="lightgray",
                 linewidth=1,
                 model=Model.POINCARE,
                 xlim=None,
                 ylim=None):

        if ax is None or fig is None:
            fig, ax = plt.subplots(figsize=(figsize, figsize))

        default_x, default_y = default_model_limits(model)

        self.xlim, self.ylim = xlim, ylim
        if xlim is None:
            self.xlim = default_x
        if ylim is None:
            self.ylim = default_y

        self.ax, self.fig = ax, fig

        plt.tight_layout()
        self.ax.axis("off")
        self.ax.set_aspect("equal")
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)

        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.linewidth = linewidth

        self.model = model

    def draw_plane(self, **kwargs):
        if self.model == Model.POINCARE or self.model == Model.KLEIN:
            plane = Circle((0., 0.), 1.0, facecolor=self.facecolor,
                           edgecolor=self.edgecolor,
                           linewidth=self.linewidth, zorder=0, **kwargs)

            self.ax.add_patch(plane)
        elif self.model == Model.HALFSPACE:
            xmin, xmax = self.xlim
            ymin, ymax = self.ylim
            plane = Rectangle((xmin, 0.), xmax - xmin, ymax,
                              facecolor=self.facecolor,
                              edgecolor=None,
                              zorder=0,
                              **kwargs)
            self.ax.add_patch(plane)

            #plot boundary
            plt.plot((xmin, xmax), (0., 0.), color=self.edgecolor,
                     linewidth=self.linewidth)

        else:
            raise DrawingError(
                "Drawing in model '{}' is not implemented".format(self.model)
            )
    def draw_geodesic(self, segment,
                      radius_threshold=RADIUS_THRESHOLD, **kwargs):
        seglist = segment.flatten_to_unit()
        default_kwargs = {
            "color":"black",
            "linewidth":1
        }
        for key, value in kwargs.items():
            default_kwargs[key] = value

        if self.model == Model.KLEIN:
            lines = LineCollection(seglist.endpoint_coords(self.model),
                                   **default_kwargs)
            self.ax.add_collection(lines)

        elif self.model == Model.POINCARE or self.model == Model.HALFSPACE:
            centers, radii, thetas = seglist.circle_parameters(model=self.model,
                                                               degrees=True)
            endpt_coords = seglist.get_endpoints().coords(self.model)

            for center, radius, theta, endpts in zip(centers, radii,
                                                     thetas, endpt_coords):
                if radius < radius_threshold:
                    arc = Arc(center, radius * 2, radius * 2,
                              theta1=theta[0], theta2=theta[1],
                              **kwargs)
                    self.ax.add_patch(arc)
                else:
                    x,y = endpts.T
                    self.ax.plot(x, y, **default_kwargs)
        else:
            raise DrawingError(
                "Drawing geodesics in model '{}' is not implemented".format(
                    self.model)
            )

    def draw_point(self, point, **kwargs):
        pointlist = point.flatten_to_unit()
        default_kwargs = {
            "color" : "black",
            "marker": "o",
            "linestyle":"none"
        }
        for key, value in kwargs.items():
            default_kwargs[key] = value

        x, y = pointlist.coords(self.model).T
        plt.plot(x, y, **default_kwargs)

    def get_circle_arcpath(self, center, radius, theta):
        """Get a matplotlib path object for the circular arc representing this
        geometric object.

        """
        cx, cy = center
        transform = Affine2D()
        transform.scale(radius)
        transform.translate(cx, cy)
        return transform.transform_path(Path.arc(theta[0], theta[1]))

    def get_straight_arcpath(self, segment):
        vertices = segment.endpoint_coords(model=self.model)
        codes = [Path.MOVETO, Path.LINETO]
        return Path(vertices, codes)

    def get_polygon_arcpath(self, polygon,
                            radius_threshold=RADIUS_THRESHOLD,
                            distance_threshold=DISTANCE_THRESHOLD):

        vertices = np.zeros((0, 2))
        codes = np.zeros((0,))
        first_segment = True

        polysegs = polygon.get_edges()
        centers, radii, thetas = polysegs.circle_parameters(model=self.model)

        for center, radius, theta, segment in zip(centers, radii, thetas, polysegs):
            if not np.isnan(radius) and radius < radius_threshold:
                g_path = self.get_circle_arcpath(center, radius, theta)
            else:
                g_path = self.get_straight_arcpath(segment)

            g_verts = g_path.vertices
            p1, p2 = segment.get_end_pair(as_points=True)

            if np.linalg.norm(p1.coords(self.model) - g_verts[0]) > distance_threshold:
                g_verts = g_verts[::-1]

            g_codes = copy.deepcopy(g_path.codes)
            if not first_segment:
                g_codes[0] = Path.LINETO

            vertices = np.concatenate((vertices, g_verts), axis=-2)
            codes = np.concatenate((codes, g_codes))
            first_segment = False

        return Path(vertices, codes)

    def draw_polygon(self, polygon, **kwargs):
        default_kwargs = {
            "facecolor": "none",
            "edgecolor": "black"
        }
        for key, value in kwargs.items():
            default_kwargs[key] = value

        polylist = polygon.flatten_to_unit()

        if self.model == Model.KLEIN:
            polys = PolyCollection(polylist.coords("klein"), **default_kwargs)
            self.ax.add_collection(polys)

        elif self.model == Model.POINCARE or self.model == Model.HALFSPACE:
            for poly in polylist:
                path = self.get_polygon_arcpath(poly)
                self.ax.add_patch(PathPatch(path, **default_kwargs))
        else:
            raise DrawingError(
                "Drawing polygons in model '{}' is not implemented".format(
                    self.model)
            )

    def draw_horosphere(self, horoball, **kwargs):
        default_kwargs = {
            "facecolor": "none",
            "edgecolor": "black"
        }
        for key, value in kwargs.items():
            default_kwargs[key] = value

        horolist = horoball.flatten_to_unit()
        if self.model == Model.POINCARE or self.model == Model.HALFSPACE:
            center, radius = horolist.sphere_parameters(model=self.model)

            circle_ctrs = center[radius<RADIUS_THRESHOLD]
            circle_radii = radius[radius<RADIUS_THRESHOLD]

            self.ax.add_collection(EllipseCollection(circle_radii * 2, circle_radii * 2,
                                                     0, units="xy",
                                                     offsets=circle_ctrs,
                                                     transOffset=self.ax.transData,
                                                     **default_kwargs))

            #TODO: draw lines for horospheres in halfspace model


        else:
            raise DrawingError(
                "Drawing horospheres in model '{}' is not implemented.".format(
                self.model)
            )

    def draw_horoarc(self, horoarc, **kwargs):
        default_kwargs = {
            "facecolor": "none",
            "edgecolor": "black"
        }
        for key, value in kwargs.items():
            default_kwargs[key] = value

        if self.model != Model.POINCARE:
            raise DrawingError(
                "Drawing horoarcs in model '{}' is not implemented.".format(
                    self.model)
            )

        horolist = horoarc.flatten_to_unit()
        centers, radii, thetas = horolist.circle_parameters(model=self.model)

        #in the half-plane model this will involve radius checking
        for center, radius, theta in zip(centers, radii, thetas):
            arc = Arc(center, radius * 2, radius * 2,
                      theta1=theta[0], theta2=theta[1],
                      **kwargs)
            self.ax.add_patch(arc)
