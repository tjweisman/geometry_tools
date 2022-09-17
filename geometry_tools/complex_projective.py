"""Work with objects in CP^1, the complex projective line, in
numerical coordinates.

"""

from copy import copy

import numpy as np

from geometry_tools import utils
from geometry_tools import projective

class CP1Object(projective.ProjectiveObject):
    def real_affine_coords(self):
        return np.squeeze(utils.c_to_r(self.affine_coords()), axis=-2)

class CP1Point(projective.Point, CP1Object):
    def __init__(self, point, coords="projective"):
        self.unit_ndims = 1
        self.aux_ndims = 0
        self.dual_ndims = 0

        try:
            self._construct_from_object(point)
            return
        except TypeError:
            pass

        pdata = np.array(point)
        # this might end up being rewritten to work the same way that
        # hyperbolic models do.
        if coords == "cx_affine":
            cx_aff_data = np.expand_dims(pdata, axis=-1)
            projective.Point.__init__(self, cx_aff_data, chart_index=0)
        elif coords == "real_affine":
            cx_aff_data = np.expand_dims(utils.r_to_c(pdata), axis=-1)
            projective.Point.__init__(self, cx_aff_data, chart_index=0)
        elif coords == "spherical":
            proj_data = spherical_to_projective(pdata)
            projective.Point.__init__(self, proj_data)
        else:
            projective.Point.__init__(self, pdata)

    def spherical_coords(self, spherical_data=None):
        if spherical_data is not None:
            self.set(spherical_to_projective(spherical_data))

        return projective_to_spherical(self.proj_data,
                                       column_vectors=False)
    def to_standard_triple(self):

        return to_standard_triple(self)

class CP1Disk(CP1Object):
    def _compute_proj_data(self, center, rad, radius_metric="affine"):
        # given the center/radius of a disk in affine coordinates,
        # compute projective coordinates of three points on the boundary of the disk
        # and the center.
        if radius_metric == "affine":
            center_coords = center.real_affine_coords()

            with np.errstate(divide="ignore", invalid="ignore"):
                normed_ctr = utils.normalize(center_coords)
                # we can just pick something arbitrary (unit-length)
                # if we're at the origin
                normed_ctr[utils.normsq(center_coords) == 0] = np.array([1.0, 0.0])

            perp_ctr = np.stack([-1 * normed_ctr[..., 1], normed_ctr[..., 0]],
                                axis=-1)

            e_rad = np.expand_dims(rad, axis=-1)

            affine_real_pts = np.stack([center_coords + e_rad * normed_ctr,
                                        center_coords - e_rad * normed_ctr,
                                        center_coords + e_rad * perp_ctr,
                                        center_coords], axis=-2)

            affine_c_pts = utils.r_to_c(affine_real_pts)
            return np.stack([np.ones_like(affine_c_pts), affine_c_pts],
                            axis=-1)
        elif radius_metric == "fs":
            center_coords = center.spherical_coords()

            q, r = np.linalg.qr(np.expand_dims(center_coords, axis=-1),
                                mode='complete')

            p1_t = np.stack([np.cos(2 * rad),
                             np.sin(2 * rad),
                             np.zeros_like(rad)], axis=-1)
            p2_t = np.stack([np.cos(2 * rad),
                             -np.sin(2 * rad),
                             np.zeros_like(rad)], axis=-1)
            p3_t = np.stack([np.cos(2 * rad),
                             np.zeros_like(rad),
                             np.sin(2 * rad)], axis=-1)

            t_pts = np.stack([p1_t, p2_t, p3_t], axis=-1)
            sph_pts = (np.expand_dims(r[..., 0, 0], axis=(-1, -2)) *
                       q @ t_pts)

            proj_pts = spherical_to_projective(
                sph_pts.swapaxes(-1, -2)
            )

            return np.concatenate([proj_pts,
                                   np.expand_dims(center.projective_coords(),
                                                  axis=-2)], axis=-2)


    def __init__(self, center, rad=None, radius_metric="affine",
                 center_coords="cx_affine"):
        if rad is None:
            CP1Object.__init__(self, center, unit_ndims=2)
            return

        ct_point = CP1Point(center, coords=center_coords)
        proj_data = self._compute_proj_data(ct_point, rad,
                                            radius_metric=radius_metric)

        CP1Object.__init__(self, proj_data, unit_ndims=2)

    def boundary_points(self):
        return CP1Point(self.proj_data[..., :3, :])

    def interior_point(self):
        return CP1Point(self.proj_data[..., -1, :])

    def circle_parameters(self):
        bdry_pts = self.boundary_points()
        bdry_aff_coords = bdry_pts.real_affine_coords()

        # there's a more pythonic way to do this but IDGAF
        p1, p2, p3 = (bdry_aff_coords[..., i, :]
                      for i in range(3))

        return utils.circle_through(p1, p2, p3)

    def center_inside(self):
        circ_ctr, circ_rad = self.circle_parameters()
        ipts = self.interior_point()

        center_affine = ipts.in_affine_chart(0)
        int_pt_coords = ipts[center_affine].real_affine_coords()
        dist_sq = utils.normsq(circ_ctr[center_affine] - int_pt_coords)

        res = np.full(self.shape, False)

        res[center_affine] = dist_sq < circ_rad[center_affine]**2

        return res

    def fs_diameter(self):
        center, radius = self.circle_parameters()
        center_norm = np.linalg.norm(center, axis=-1)

        res = np.arctan(center_norm + radius) - np.arctan(center_norm - radius)
        inverted = ~self.center_inside()
        res[inverted] = np.pi - res[inverted]

        return res

    def fs_center(self):
        center, radius = self.circle_parameters()
        center_norm = np.linalg.norm(center, axis=-1)

        fs_ctr_dist = np.tan(
            (np.arctan(center_norm + radius) +
             np.arctan(center_norm - radius)) / 2
        )

        scale_factor = np.ones_like(center_norm)
        nonzero_scale = np.abs(fs_ctr_dist) > 0
        scale_factor[nonzero_scale] = (fs_ctr_dist[nonzero_scale] /
                                       center_norm[nonzero_scale])

        fs_center = CP1Point(np.expand_dims(scale_factor, axis=-1) * center,
                             coords="real_affine")

        # apply a sphere inversion for disks containing infinity
        inverted = ~self.center_inside()
        inverted_pts = inversion() @ fs_center[inverted]
        fs_center[inverted] = inverted_pts

        fs_center.proj_data[inverted] = np.conjugate(
            fs_center.proj_data[inverted]
        )

        return fs_center

    def inversion(self):
        basechange = self.boundary_points().to_standard_triple()
        std_inversion = projective.Transformation(
            np.array([[1.0, 0.], [0., -1.]])
        )
        return basechange.inv() @ std_inversion @ basechange

    def complement(self):
        new_int = self.inversion() @ self.interior_point()

        complement = CP1Disk(copy(self.proj_data))
        complement.proj_data[..., -1, :] = new_int.proj_data
        return complement

    def contains(self, other, broadcast="elementwise"):
        if broadcast == "pairwise":
            s_to_use = self.flatten_to_unit()
            o_to_use = other.flatten_to_unit()
        else:
            s_to_use = self
            o_to_use = other

        s_aff = s_to_use.center_inside()
        o_aff = o_to_use.center_inside()

        sctr, srad = s_to_use.circle_parameters()
        octr, orad = o_to_use.circle_parameters()

        contain, contained, intersect = utils.disk_interactions(
            sctr, srad, octr, orad, broadcast=broadcast
        )

        res = np.full(contain.shape, False)

        if broadcast == "elementwise":
            res[s_aff & o_aff] = contain[s_aff & o_aff]
            res[~s_aff & o_aff] = ~intersect[~s_aff & o_aff]
            res[~s_aff & ~o_aff] = contained[~s_aff & ~o_aff]
            return res

        affaffmask = np.logical_and(
            np.expand_dims(s_aff, axis=1),
            np.expand_dims(o_aff, axis=0)
        )
        naffaffmask = np.logical_and(
            np.expand_dims(~s_aff, axis=1),
            np.expand_dims(o_aff, axis=0)
        )
        naffnaffmask = np.logical_and(
            np.expand_dims(~s_aff, axis=1),
            np.expand_dims(~o_aff, axis=0)
        )
        np.putmask(res, affaffmask, contain)
        np.putmask(res, naffaffmask, ~intersect)
        np.putmask(res, naffnaffmask, contained)
        return res

    def intersects(self, other, broadcast="elementwise"):
        if broadcast == "pairwise":
            s_to_use = self.flatten_to_unit()
            o_to_use = other.flatten_to_unit()
        else:
            s_to_use = self
            o_to_use = other

        s_aff = s_to_use.center_inside()
        o_aff = o_to_use.center_inside()

        sctr, srad = s_to_use.circle_parameters()
        octr, orad = o_to_use.circle_parameters()

        contain, contained, intersect = utils.disk_interactions(
            sctr, srad, octr, orad, broadcast=broadcast
        )

        res = np.full(contain.shape, True)

        if broadcast == "elementwise":
            res[s_aff & o_aff] = intersect[s_aff & o_aff]
            res[~s_aff & o_aff] = ~contain[~s_aff & o_aff]
            res[s_aff & ~o_aff] = ~contained[~s_aff & ~o_aff]
            return res

        affaffmask = np.logical_and(
            np.expand_dims(s_aff, axis=1),
            np.expand_dims(o_aff, axis=0)
        )
        naffaffmask = np.logical_and(
            np.expand_dims(~s_aff, axis=1),
            np.expand_dims(o_aff, axis=0)
        )
        affnaffmask = np.logical_and(
            np.expand_dims(s_aff, axis=1),
            np.expand_dims(~o_aff, axis=0)
        )
        np.putmask(res, affaffmask, intersect)
        np.putmask(res, naffaffmask, ~contain)
        np.putmask(res, affnaffmask, ~contained)
        return res

def projective_to_spherical(points, column_vectors=False):
    ppoints = np.array(points)
    if column_vectors:
        ppoints = spoints.swapaxes(-1, -2)

    z0 = ppoints[..., 0]
    z1 = ppoints[..., 1]
    normsq = np.abs(z0 * np.conjugate(z0) + z1 * np.conjugate(z1))
    horizontal = utils.c_to_r(2 * np.conjugate(z0) * z1 / normsq)
    vertical = (np.abs(z1 * np.conjugate(z1)) -
                np.abs(z0 * np.conjugate(z0))) / normsq

    spherical = np.concatenate([horizontal, np.expand_dims(vertical, axis=-1)],
                               axis=-1)

    if column_vectors:
        spherical = spherical.swapaxes(-1, -2)

    return spherical

def spherical_to_projective(points, column_vectors=False):
    spoints = np.array(points)
    if column_vectors:
        spoints = spoints.swapaxes(-1, -2)

    chart2 = spoints[..., -1] > 0

    res = np.stack([1 - spoints[..., -1],
                    utils.r_to_c(spoints[..., :-1])],
                   axis=-1)

    # need to do this in two charts
    res[chart2] = np.stack(
        [np.conjugate(utils.r_to_c(spoints[chart2, :-1])),
         1 + spoints[chart2, -1]],
        axis=-1)

    if column_vectors:
        res = res.swapaxes(-1, -2)

    return res

def to_standard_triple(triple):
    tpoint = CP1Point(triple)
    if tpoint.proj_data.shape[-2] != 3:
        raise GeometryError(
            f"Cannot take an array of {self.proj_data.shape[-2]}-tuples"
            " to an array of triples."
        )

    tdata = tpoint.projective_coords()

    res = np.linalg.inv(tdata[..., :2, :])
    p3_t = np.expand_dims(tdata[..., 2, :],
                          axis=-2) @ res

    eigenvalue = np.emath.sqrt(p3_t[..., 1] / p3_t[..., 0])

    res[..., 0] *= eigenvalue
    res[..., 1] /= eigenvalue

    return projective.Transformation(res)

def inversion():
    return projective.Transformation(
        np.array([[0, -1],[1,0]])
    )
