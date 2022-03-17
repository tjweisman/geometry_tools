"""geometry_tools.drawtools

Provides some useful functions for turning objects from the hyperbolic
module into matplotlib figures.

"""

import copy

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc, PathPatch
from matplotlib.collections import LineCollection, PolyCollection

from matplotlib.transforms import Affine2D
from matplotlib.path import Path

from geometry_tools import hyperbolic, utils

#TODO: make a robust way to handle different coordinate systems
KLEIN_NAMES = ["klein", "kleinian"]

#I played around with this a bit, but it's an eyeball test
#TBH. Determines the radius at which we start approximating circular
#arcs with straight lines.
RADIUS_THRESHOLD = 80

#how far apart points can be before we decide that we ordered the
#polygon wrong
DISTANCE_THRESHOLD = 1e-5

class HyperbolicDrawing:
    def __init__(self, figsize=8,
                 ax=None,
                 fig=None,
                 facecolor="lightgray",
                 edgecolor="black",
                 linewidth=1,
                 model="poincare"):
        if ax is None or fig is None:
            fig, ax = plt.subplots(figsize=(figsize, figsize))

        self.ax, self.fig = ax, fig

        plt.tight_layout()
        self.ax.axis("off")
        self.ax.set_aspect("equal")
        self.ax.set_xlim((-1.1, 1.1))
        self.ax.set_ylim((-1.1, 1.1))

        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.linewidth = linewidth

        self.model = model

    def draw_plane(self, **kwargs):
        plane = Circle((0., 0.), 1.0, facecolor=self.facecolor,
                       edgecolor=self.edgecolor,
                       linewidth=self.linewidth, zorder=0, **kwargs)

        self.ax.add_patch(plane)

    def draw_geodesic(self, segment,
                      radius_threshold=RADIUS_THRESHOLD, **kwargs):
        seglist = segment.flatten_to_unit()
        default_kwargs = {
            "color":"black",
            "linewidth":1
        }
        for key, value in kwargs.items():
            default_kwargs[key] = value

        if self.model == "klein":
            lines = LineCollection(seglist.get_endpoints().kleinian_coords(),
                                   **default_kwargs)
            self.ax.add_collection(lines)

        elif self.model == "poincare":
            centers, radii, thetas = seglist.circle_parameters(degrees=True)
            endpt_coords = seglist.get_endpoints().poincare_coords()

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

    def draw_point(self, point, **kwargs):
        pointlist = point.flatten_to_unit()
        default_kwargs = {
            "color" : "black",
            "marker": "o",
            "linestyle":"none"
        }
        for key, value in kwargs.items():
            default_kwargs[key] = value

        if self.model == "klein":
            x, y = pointlist.kleinian_coords().T

        elif self.model == "poincare":
            x, y = pointlist.poincare_coords().T

        plt.plot(x, y, **default_kwargs)

    def get_circle_arcpath(self, circle_params,
                           radius_threshold=RADIUS_THRESHOLD):
        """Get a matplotlib path object for the circular arc representing this
        geometric object.

        """
        center, radius, theta = circle_params
        if not np.isnan(radius) and radius < radius_threshold:
            cx, cy = center
            transform = Affine2D()
            transform.scale(radius)
            transform.translate(cx, cy)
            return transform.transform_path(Path.arc(theta[0], theta[1]))

        vertices = arc.endpoint_coords(coords=self.model)
        codes = [Path.MOVETO, Path.LINETO]
        return Path(vertices, codes)

    def get_polygon_arcpath(self, polygon,
                            radius_threshold=RADIUS_THRESHOLD,
                            distance_threshold=DISTANCE_THRESHOLD):

        vertices = np.zeros((0, 2))
        codes = np.zeros((0,))
        first_segment = True

        polysegs = polygon.get_edges()
        centers, radii, thetas = polysegs.circle_parameters(coords=self.model)

        for center, radius, theta, segment in zip(centers, radii, thetas, polysegs):
            g_path = self.get_circle_arcpath((center, radius, theta),
                                             radius_threshold)
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

        if self.model == "klein":
            polys = PolyCollection(polylist.coords("klein"), **default_kwargs)
            self.ax.add_collection(polys)

        elif self.model == "poincare":
            for poly in polylist:
                path = self.get_polygon_arcpath(poly)
                self.ax.add_patch(PathPatch(path, **default_kwargs))

    def draw_horoball(self, horoball, **kwargs):
        pass
