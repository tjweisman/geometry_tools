"""geometry_tools.drawtools

Provides some useful functions for turning objects from the hyperbolic
module into matplotlib figures.

"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc
from matplotlib.collections import LineCollection

from geometry_tools import hyperbolic, utils

RADIUS_THRESHOLD = 100

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
                       linewidth=self.linewidth, **kwargs)

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

    def draw_polygon(self, polygon, **kwargs):
        pass

    def draw_horoball(self, horoball, **kwargs):
        pass
