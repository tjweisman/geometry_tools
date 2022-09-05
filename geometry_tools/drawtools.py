"""This submodule provides an interface between the
`geometry_tools.projective` and `geometry_tools.hyperbolic` submodules
and [matplotlib](https://matplotlib.org/).

The central classes in this module are `ProjectiveDrawing` and
`HyperbolicDrawing`. To create a matplotlib figure, instantiate one of
these classes and use the provided methods to add geometric objects to
the drawing.

```python
from geometry_tools import hyperbolic, drawtools
from numpy import pi

hyp_drawing = drawtools.HyperbolicDrawing(model="halfplane")
triangle = hyperbolic.Polygon.regular_polygon(3, angle=pi / 6)

hyp_drawing.draw_plane()
hyp_drawing.draw_polygon(triangle, facecolor="lightgreen")

hyp_drawing.show()
```

![triangle](triangle.png)

    """

import copy

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc, PathPatch, Rectangle, Polygon
from matplotlib.collections import LineCollection, PolyCollection, EllipseCollection

from matplotlib.transforms import Affine2D
from matplotlib.path import Path

from geometry_tools import hyperbolic, utils, projective
from geometry_tools.hyperbolic import Model

#I played around with this a bit, but it's an eyeball test
#TBH. Determines the radius at which we start approximating circular
#arcs with straight lines.
RADIUS_THRESHOLD = 80

#how far apart points can be before we decide that we ordered the
#polygon wrong
DISTANCE_THRESHOLD = 1e-4

#the default amount of "room" we leave outside the boundary of our model
DRAW_NEIGHBORHOOD = 0.1

#when drawing objects "to infinity", how far offscreen we draw them
#(as a % of the width/height)
OFFSCREEN_FACTOR = 0.1

#this is a bit unpythonic since these are meant to be constants
def default_model_limits(model):
    if model == Model.POINCARE or model == Model.KLEIN:
        return ((-1 - DRAW_NEIGHBORHOOD, 1 + DRAW_NEIGHBORHOOD),
                (-1 - DRAW_NEIGHBORHOOD, 1 + DRAW_NEIGHBORHOOD))

    if model == Model.HALFSPACE:
        return ((-6., 6.),
                (-1 * DRAW_NEIGHBORHOOD, 8.))

class DrawingError(Exception):
    """Thrown if we try and draw an object in a model which we haven't
    implemented yet.

    """
    pass

class ProjectiveDrawing:
    def __init__(self, figsize=8,
                 ax=None,
                 fig=None,
                 xlim=(-5., 5.),
                 ylim=(-5., 5.),
                 transform=None):

        if ax is None or fig is None:
            fig, ax = plt.subplots(figsize=(figsize, figsize))

        self.xlim, self.ylim = xlim, ylim

        self.width = self.xlim[1] - self.xlim[0]
        self.height = self.ylim[1] - self.ylim[0]

        self.ax, self.fig = ax, fig

        plt.tight_layout()
        self.ax.axis("off")
        self.ax.set_aspect("equal")
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)

        self.transform = projective.identity(2)
        if transform is not None:
            self.transform = transform

    def draw_point(self, point, **kwargs):
        pointlist = self.transform @ point.flatten_to_unit()
        default_kwargs = {
            "color" : "black",
            "marker": "o",
            "linestyle":"none"
        }
        for key, value in kwargs.items():
            default_kwargs[key] = value

        x, y = pointlist.affine_coords().T
        plt.plot(x, y, **default_kwargs)

    def draw_proj_segment(self, segment, **kwargs):
        seglist = self.transform @ segment.flatten_to_unit()
        default_kwargs = {
            "color":"black",
            "linewidth":1
        }
        for key, value in kwargs.items():
            default_kwargs[key] = value

        lines = LineCollection(seglist.endpoint_affine_coords(),
                               **default_kwargs)
        self.ax.add_collection(lines)

    def view_diam(self):
        return np.sqrt(self.width * self.width + self.height * self.height)

    def view_ctr(self):
        return np.array([(self.xlim[0] + self.xlim[1])/2,
                         (self.ylim[0] + self.ylim[1])/2])

    def nonaff_poly_path(self, polygon):
        pass

    def draw_nonaff_polygon(self, polygon, **kwargs):
        if len(polygon.proj_data) == 0:
            return

         # first, find the first index where we switch signs
        sign_switch = utils.first_sign_switch(polygon.projective_coords()[..., 0])

        # roll the coordinates by the signs
        coord_mat = polygon.projective_coords()

        rows, cols = np.ogrid[:coord_mat.shape[0], :coord_mat.shape[1]]
        cols = (cols + sign_switch[:, np.newaxis]) % coord_mat.shape[-2]
        rolled_coords = coord_mat[rows, cols]

        # find the index where signs switch back
        second_switch = utils.first_sign_switch(rolled_coords[..., 0])

        # re-index polygon affine coordinates by first sign switch
        rolled_polys = projective.Polygon(rolled_coords)
        poly_affine = rolled_polys.affine_coords()

        # find affine coordinates of sign-switch points
        s1_v1 = poly_affine[..., -1, :]
        s1_v2 = poly_affine[..., 0, :]

        s2_v1 = np.take_along_axis(poly_affine, second_switch[:, np.newaxis, np.newaxis] - 1, axis=1
                                  ).squeeze(axis=1)
        s2_v2 = np.take_along_axis(poly_affine, second_switch[:, np.newaxis, np.newaxis], axis=1
                                  ).squeeze(axis=1)

        # compute normalized (affine) displacement vectors between
        # endpoints of segments which cross infinity
        disp_1 = s1_v2 - s1_v1
        disp_2 = s2_v2 - s2_v1

        n_disp_1 = utils.normalize(disp_1)
        n_disp_2 = utils.normalize(disp_2)

        # compute dummy vertex coordinates for segments which cross infinity.
        # this could be DRYer.
        dummy_p1v1 = s1_v2 + (
            n_disp_1 * (self.view_diam() + np.linalg.norm(s1_v2, axis=-1))[:, np.newaxis]
        )
        dummy_p2v1 = s1_v1 - (
            n_disp_1 * (self.view_diam() + np.linalg.norm(s1_v1, axis=-1))[:, np.newaxis]
        )

        dummy_p1v2 = s2_v1 - (
            n_disp_2 * (self.view_diam() + np.linalg.norm(s2_v1, axis=-1))[:, np.newaxis]
        )
        dummy_p2v2 = s2_v2 + (
            n_disp_2 * (self.view_diam() + np.linalg.norm(s2_v2, axis=-1))[:, np.newaxis]
        )

        dummy_coords_1 = np.stack([dummy_p1v2, dummy_p1v1], axis=-2)
        dummy_coords_2 = np.stack([dummy_p2v1, dummy_p2v2], axis=-2)


        # draw a pair of polygons for each non-affine polygon
        for poly_coords, s_index, dc_1, dc_2 in zip(
            poly_affine, second_switch, dummy_coords_1, dummy_coords_2):

            p1 = Polygon(np.concatenate([poly_coords[:s_index], dc_1]),
                        **kwargs)
            p2 = Polygon(np.concatenate([poly_coords[s_index:], dc_2]),
                        **kwargs)
            self.ax.add_patch(p1)
            self.ax.add_patch(p2)



    def draw_polygon(self, polygon, assume_affine=True, **kwargs):
        default_kwargs = {
            "facecolor": "none",
            "edgecolor": "black"
        }
        for key, value in kwargs.items():
            default_kwargs[key] = value

        polylist = self.transform @ polygon.flatten_to_unit()

        if assume_affine:
            polys = PolyCollection(polylist.affine_coords(), **default_kwargs)
            self.ax.add_collection(polys)
            return

        in_chart = polylist.in_standard_chart()
        affine_polys = PolyCollection(polylist[in_chart].affine_coords(),
                                      **default_kwargs)
        self.ax.add_collection(affine_polys)

        self.draw_nonaff_polygon(polylist[~in_chart], **default_kwargs)

    def set_transform(self, transform):
        self.transform = transform

    def add_transform(self, transform):
        self.transform = transform @ self.transform

    def precompose_transform(self, transform):
        self.transform = self.transform @ transform

    def show(self):
        plt.show()

class HyperbolicDrawing(ProjectiveDrawing):
    def __init__(self, figsize=8,
                 ax=None,
                 fig=None,
                 facecolor="aliceblue",
                 edgecolor="lightgray",
                 linewidth=1,
                 model=Model.POINCARE,
                 xlim=None,
                 ylim=None,
                 transform=None):

        if ax is None or fig is None:
            fig, ax = plt.subplots(figsize=(figsize, figsize))

        default_x, default_y = default_model_limits(model)

        self.xlim, self.ylim = xlim, ylim
        if xlim is None:
            self.xlim = default_x
        if ylim is None:
            self.ylim = default_y

        self.width = self.xlim[1] - self.xlim[0]
        self.height = self.ylim[1] - self.ylim[0]

        self.left_infinity = self.xlim[0] - OFFSCREEN_FACTOR * self.width
        self.right_infinity = self.xlim[1] + OFFSCREEN_FACTOR * self.width
        self.up_infinity = self.ylim[1] + OFFSCREEN_FACTOR * self.height
        self.down_infinity = self.ylim[0] - OFFSCREEN_FACTOR * self.height
        self.h_infinity = self.right_infinity - self.left_infinity
        self.v_infinity = self.up_infinity - self.down_infinity

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

        self.transform = hyperbolic.identity(2)

        if transform is not None:
            self.transform = transform



    def draw_plane(self, **kwargs):
        default_kwargs = {
            "facecolor": self.facecolor,
            "edgecolor": self.edgecolor,
            "linewidth": self.linewidth,
            "zorder": 0
        }
        for key, value in kwargs.items():
            default_kwargs[key] = value

        if self.model == Model.POINCARE or self.model == Model.KLEIN:
            plane = Circle((0., 0.), 1.0, **default_kwargs)

            self.ax.add_patch(plane)
        elif self.model == Model.HALFSPACE:
            xmin, xmax = self.xlim
            ymin, ymax = self.ylim
            plane = Rectangle((self.left_infinity, 0.),
                              self.h_infinity, self.up_infinity,
                              facecolor=self.facecolor,
                              edgecolor=self.edgecolor,
                              zorder=0,
                              **kwargs)
            self.ax.add_patch(plane)

        else:
            raise DrawingError(
                "Drawing in model '{}' is not implemented".format(self.model)
            )

    def get_vertical_segment(self, endpts):
        ordered_endpts = endpts[:]
        if (np.isnan(endpts[0,0]) or
            endpts[0, 0] < self.left_infinity or
            endpts[0, 0] > self.right_infinity):
            ordered_endpts = np.flip(endpts, axis=0)

        if (np.isnan(ordered_endpts[1, 0]) or
            ordered_endpts[1, 0] < self.left_infinity or
            ordered_endpts[1, 0] > self.right_infinity):

            ordered_endpts[1, 1] = self.up_infinity

        ordered_endpts[1, 0] = ordered_endpts[0, 0]

        return ordered_endpts



    def draw_geodesic(self, segment,
                      radius_threshold=RADIUS_THRESHOLD, **kwargs):
        seglist = self.transform @ segment.flatten_to_unit()
        default_kwargs = {
            "color":"black",
            "linewidth":1
        }
        for key, value in kwargs.items():
            default_kwargs[key] = value

        if self.model not in [Model.KLEIN, Model.POINCARE, Model.HALFSPACE]:
            raise DrawingError(
                "Drawing geodesics in model '{}' is not implemented".format(
                    self.model)
            )

        if self.model == Model.KLEIN:
            lines = LineCollection(seglist.endpoint_coords(self.model),
                                   **default_kwargs)
            self.ax.add_collection(lines)
            return

        centers, radii, thetas = seglist.circle_parameters(model=self.model,
                                                               degrees=True)
        for center, radius, theta, segment in zip(centers, radii,
                                                  thetas, seglist):
            if np.isnan(radius) or radius > radius_threshold:
                arcpath = self.get_straight_arcpath(segment)
                arc = PathPatch(arcpath, **default_kwargs)
                self.ax.add_patch(arc)
                continue

            arc = Arc(center, radius * 2, radius * 2,
                      theta1=theta[0], theta2=theta[1],
                      **kwargs)
            self.ax.add_patch(arc)


    def draw_point(self, point, **kwargs):
        pointlist = self.transform @ point.flatten_to_unit()
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
        endpts = segment.endpoint_coords(self.model)

        if self.model == Model.POINCARE:
            return Path(endpts, [Path.MOVETO, Path.LINETO])
        if self.model == Model.HALFSPACE:
            v_endpts = self.get_vertical_segment(endpts)
            return Path(v_endpts, [Path.MOVETO, Path.LINETO])

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

            p1_opp_dist = np.linalg.norm(p1.coords(self.model) - g_verts[-1])
            p2_opp_dist = np.linalg.norm(p2.coords(self.model) - g_verts[0])
            if (p1_opp_dist < distance_threshold or
                p2_opp_dist < distance_threshold):
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

        polylist = self.transform @ polygon.flatten_to_unit()

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

        horolist = self.transform @ horoball.flatten_to_unit()
        if self.model == Model.POINCARE or self.model == Model.HALFSPACE:
            center, radius = horolist.sphere_parameters(model=self.model)

            good_indices = ((radius < RADIUS_THRESHOLD) &
                            ~np.isnan(radius) &
                            ~(np.isnan(center).any(axis=-1)))

            circle_ctrs = center[good_indices]
            circle_radii = radius[good_indices]

            if len(circle_ctrs) > 0:
                self.ax.add_collection(
                    EllipseCollection(circle_radii * 2, circle_radii * 2,
                                      0, units="xy", offsets=circle_ctrs,
                                      transOffset=self.ax.transData,
                                      **default_kwargs)
                )

            if self.model == Model.HALFSPACE:
                for horoball in horolist[~good_indices]:
                    height = horoball.ref_coords(self.model)[1]
                    h_rect = Rectangle((self.left_infinity, height),
                                       self.h_infinity,
                                       self.up_infinity - height,
                                       **default_kwargs)

                    self.ax.add_patch(h_rect)
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

        if self.model != Model.POINCARE and self.model != Model.HALFSPACE:
            raise DrawingError(
                "Drawing horoarcs in model '{}' is not implemented.".format(
                    self.model)
            )

        horolist = self.transform @ horoarc.flatten_to_unit()
        endpts = horolist.endpoint_coords(model=self.model)
        centers, radii, thetas = horolist.circle_parameters(model=self.model)

        for center, radius, theta, endpt in zip(centers, radii, thetas, endpts):
            if np.isnan(radius) or radius > RADIUS_THRESHOLD:
                path = Path(endpt, [Path.MOVETO, Path.LINETO])
                pathpatch = PathPatch(path, **default_kwargs)
                self.ax.add_patch(pathpatch)
            else:
                arc = Arc(center, radius * 2, radius * 2,
                          theta1=theta[0], theta2=theta[1],
                          **default_kwargs)
                self.ax.add_patch(arc)

    def draw_boundary_arc(self, boundary_arc, **kwargs):
        default_kwargs = {
            "edgecolor": "black",
            "linewidth": 3
        }
        for key, value in kwargs.items():
            default_kwargs[key] = value

        arclist = self.transform @ boundary_arc.flatten_to_unit()

        if self.model == Model.POINCARE or self.model == Model.KLEIN:
            centers, radii, thetas = arclist.circle_parameters(model=self.model)
            for center, radius, theta in zip(centers, radii, thetas):
                arc = Arc(center, radius * 2, radius * 2,
                          theta1=theta[0], theta2=theta[1],
                          **default_kwargs)

                self.ax.add_patch(arc)

        elif self.model == Model.HALFSPACE:
            endpoints = arclist.endpoint_coords(self.model, ordered=True)

            endpoints[..., 1] = 0.
            endpoints[np.isnan(endpoints)[..., 0], 0] = np.inf

            # first, draw all the lines where we go left to right
            leftright = (endpoints[..., 0, 0] < endpoints[..., 1, 0])
            leftright_endpts = endpoints[leftright]

            leftright_arcs = LineCollection(leftright_endpts, **default_kwargs)
            self.ax.add_collection(leftright_arcs)

            # then, draw all the lines that wrap around infinity

            infty_right = np.array([self.right_infinity, 0.])
            infty_left = np.array([self.left_infinity, 0.])

            to_right = np.broadcast_to(infty_right, endpoints[~leftright, 0].shape)
            left_to = np.broadcast_to(infty_left, endpoints[~leftright, 1].shape)

            coords1 = np.stack([endpoints[~leftright, 0], to_right], axis=-2)
            coords2 = np.stack([endpoints[~leftright, 1], left_to], axis=-2)

            right_arcs = LineCollection(coords1, **default_kwargs)
            left_arcs = LineCollection(coords2, **default_kwargs)

            self.ax.add_collection(right_arcs)
            self.ax.add_collection(left_arcs)

        else:
            raise DrawingError(
                "Drawing boundary arcs in model '{}' is not implemented.".format(
                    self.model)
            )
