"""Work with objects in CP^1, the complex projective line, in
numerical coordinates.

"""

import numpy as np

from geometry_tools import utils
from geometry_tools import projective

class CP1Object(projective.ProjectiveObject):
    def real_affine_coords(self):
        return np.squeeze(utils.c_to_r(self.affine_coords()), axis=-2)

class CP1Disk(CP1Object):
    def _compute_proj_data(self, ctr, rad):
        # given the center/radius of a disk in affine coordinates,
        # compute projective coordinates of three points on the boundary of the disk
        # and the center.

        normed_ctr = utils.normalize(ctr)
        perp_ctr = np.stack([-1 * normed_ctr[..., 1], normed_ctr[..., 0]],
                            axis=-1)

        e_rad = np.expand_dims(rad, axis=-1)

        affine_real_pts = np.stack([ctr + e_rad * normed_ctr,
                               ctr - e_rad * normed_ctr,
                               ctr + e_rad * perp_ctr,
                               ctr], axis=-2)

        affine_c_pts = utils.r_to_c(affine_real_pts)

        return np.stack([np.ones_like(affine_c_pts), affine_c_pts],
                         axis=-1)

    def __init__(self, aff_ctr, aff_rad=None):
        if aff_rad is None:
            CP1Object.__init__(self, aff_ctr, unit_ndims=2)
            return

        proj_data = self._compute_proj_data(aff_ctr, aff_rad)
        CP1Object.__init__(self, proj_data, unit_ndims=2)

    def boundary_points(self):
        return CP1Object(self.proj_data[..., :3, :])

    def interior_point(self):
        return CP1Object(self.proj_data[..., -1, :])

    def circle_parameters(self):
        bdry_pts = self.boundary_points()
        bdry_aff_coords = bdry_pts.real_affine_coords()

        # there's a more pythonic way to do this but IDGAF
        p1, p2, p3 = (bdry_aff_coords[..., i, :]
                      for i in range(3))

        return utils.circle_through(p1, p2, p3)

    def center_inside(self):
        circ_ctr, circ_rad = self.circle_parameters()
        int_pt_coords = self.interior_point().real_affine_coords()

        dist_sq = utils.normsq(circ_ctr - int_pt_coords)
        return dist_sq < circ_rad * circ_rad
