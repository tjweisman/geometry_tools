"""geometry_tools.hyperbolic

Provides classes to model objects in hyperbolic space with numerical
coordinates.

"""

from copy import copy

import numpy as np
import scipy
from scipy.optimize import fsolve

from geometry_tools import projective, representation, utils

#arbitrary
ERROR_THRESHOLD = 1e-8
BOUNDARY_THRESHOLD = 1e-5

CHECK_LIGHT_CONE = False

class GeometryError(Exception):
    """Thrown if there's an attempt to construct a hyperbolic object with
    numerical data that doesn't make sense for that type of object."""
    pass

class HyperbolicObject:
    """Model for an abstract object in hyperbolic space.

    Subclasses of HyperbolicObject model more specific objects in
    hyperbolic geometry, e.g. points, geodesic segments, hyperplanes,
    ideal points.

    Every object in hyperbolic space H^n has the underlying data of an
    ndarray whose last dimension is n+1. This is interpreted as a
    collection of (row) vectors in R^(n+1), or more precisely R^(n,1),
    which transforms via the action of the special orthogonal group
    O(n,1).

    The last unit_ndims dimensions of this underlying array represent
    a single object in hyperbolic space, which transforms as a unit
    under the action of O(n,1). The remaining dimensions of the array
    are used to represent an array of such "unit objects."

    """
    def __init__(self, space, hyp_data):
        self.unit_ndims = 1
        try:
            self._construct_from_object(hyp_data)
        except TypeError:
            self.space = space
            self.set(hyp_data)

    def _assert_geometry_valid(self, hyp_data):
        if hyp_data.ndim < self.unit_ndims:
            raise GeometryError(
                ("{} expects an array with ndim at least {}, got array of shape"
                ).format(
                    self.__class__.__name__, self.unit_ndims, hyp_data.shape
                )
            )
        if self.space.dimension + 1 != hyp_data.shape[-1]:
            raise GeometryError(
                ("Hyperbolic space has dimension {}, but received array of of shape"
                 " {}").format(self.space.dimension, hyp_data.shape)
            )

    def _construct_from_object(self, hyp_obj):
        """if we're passed a hyperbolic object or an array of hyperbolic
        objects, build a new one out of them

        """
        try:
            self.space = hyp_obj.space
            self.set(hyp_obj.hyp_data)
            return
        except AttributeError:
            pass

        try:
            # TODO: this will break if this iterable is a generator
            array = np.array([obj.hyp_data for obj in hyp_obj])
            self.space = hyp_obj[0].space
            self.set(array)
            return
        except (TypeError, AttributeError):
            pass

        raise TypeError

    def obj_shape(self):
        """Get the shape of the ndarray of "unit objects" this
        HyperbolicObject represents.

        """
        return self.hyp_data.shape[:-1 * self.unit_ndims]

    def set(self, hyp_data):
        """set the underlying data of the hyperbolic object.

        Subclasses may override this method to give special names to
        portions of the underlying data.

        """
        self._assert_geometry_valid(hyp_data)
        self.hyp_data = hyp_data

    def flatten_to_unit(self):
        """return a flattened version of the hyperbolic object.
        """
        flattened = copy(self)
        new_shape = (-1,) + self.hyp_data.shape[-1 * self.unit_ndims:]
        new_data = np.reshape(self.hyp_data, new_shape)
        flattened.set(new_data)
        return flattened

    def __repr__(self):
        return "({}, {})".format(
            self.__class__,
            self.hyp_data.__repr__()
        )

    def __str__(self):
        return "{} with data:\n{}".format(
            self.__class__.__name__, self.hyp_data.__str__()
        )

    def __getitem__(self, item):
        return self.__class__(self.space, self.hyp_data[item])

    def projective_coords(self, hyp_data=None):
        """wrapper for HyperbolicObject.set, since underlying coordinates are
        projective."""
        if hyp_data is not None:
            self.set(hyp_data)

        return self.hyp_data

    def kleinian_coords(self, hyp_data=None):
        """Get kleinian (affine) coordinates for this object.
        """
        if hyp_data is not None:
            self.set(self.space.projective.affine_to_projective(hyp_data))

        return self.space.kleinian_coords(self.hyp_data)

class Point(HyperbolicObject):
    """Model for a point (or ndarray of points) in the closure of
    hyperbolic space.

    """
    def __init__(self, space, point, coords="projective"):
        self.unit_ndims = 1
        self.space = space

        try:
            self._construct_from_object(point)
            return
        except TypeError:
            pass

        if coords == "kleinian":
            self.kleinian_coords(point)
        elif coords == "poincare":
            self.poincare_coords(point)
        else:
            self.set(point)

    def _assert_geometry_valid(self, hyp_data):
        super()._assert_geometry_valid(hyp_data)
        if not CHECK_LIGHT_CONE:
            return

        if not self.space.all_timelike(hyp_data):
            raise GeometryError( ("Point data must consist of vectors in the "
                                  "closure of the Minkowski light cone"))

    def hyperboloid_coords(self, hyp_data=None):
        """Get point coordinates on the unit hyperboloid
        """
        if hyp_data is not None:
            self.set(hyp_data)

        return self.space.hyperboloid_coords(self.hyp_data)

    def poincare_coords(self, hyp_data=None):
        """Get point coordinates in the Poincare ball (radius 1)
        """
        if hyp_data is not None:
            klein = self.space.poincare_to_kleinian(np.array(hyp_data))
            self.kleinian_coords(klein)

        return self.space.kleinian_to_poincare(self.kleinian_coords())

    def distance(self, other):
        """compute elementwise distances (with respect to last two dimensions)
        between self and other

        """
        #TODO: allow for pairwise distances
        products = utils.apply_bilinear(self.hyp_data, other.hyp_data,
                                        self.space.minkowski())

        return np.arccosh(np.abs(products))

    def origin_to(self):
        """Get an isometry taking the origin to this point.

        Not guaranteed to be orientation-preserving."""
        return self.space.timelike_to(self.hyp_data)

    def unit_tangent_towards(self, other):
        """Get the unit tangent vector pointing from this point to another
        point in hyperbolic space.

        """
        diff = other.hyp_data - self.hyp_data
        return TangentVector(self.space, self, diff).normalized()


class DualPoint(HyperbolicObject):
    """Model for a "dual point" in hyperbolic space (a point in the
    complement of the projectivization of the Minkowski light cone,
    corresponding to a geodesic hyperplane in hyperbolic space)

    """

    def _assert_geometry_valid(self, hyp_data):
        super()._assert_geometry_valid(hyp_data)
        if not CHECK_LIGHT_CONE:
            return

        if not self.space.all_spacelike(hyp_data):
            raise GeometryError("Dual point data must consist of vectors"
                                " in the complement of the Minkowski light cone")

class IdealPoint(HyperbolicObject):
    """Model for an ideal point in hyperbolic space (lying on the boundary
    of the projectivization of the Minkowski light cone)

    """
    def _assert_geometry_valid(self, hyp_data):
        super()._assert_geometry_valid(hyp_data)

        if not CHECK_LIGHT_CONE:
            return

        if not self.space.all_lightlike(hyp_data):
            raise GeometryError("Ideal point data must consist of vectors"
                                "in the boundary of the Minkowski light cone")

class Subspace(IdealPoint):
    """Model for a geodesic subspace of hyperbolic space.

    """
    def __init__(self, space, hyp_data):
        super().__init__(space, hyp_data)
        self.unit_ndims = 2

    def set(self, hyp_data):
        HyperbolicObject.set(self, hyp_data)
        self.ideal_basis = hyp_data

    def ideal_basis_coords(self, coords="kleinian"):
        """Get coordinates for a basis of this subspace lying on the ideal
        boundary of H^n.

        """
        if coords == "kleinian":
            return self.space.kleinian_coords(self.ideal_basis)

        return self.ideal_basis

    def sphere_parameters(self):
        """Get parameters describing a k-sphere corresponding to this subspace
        in the Poincare model.

        Returns a pair (center, radius). In the Poincare model, this
        subspace lies on a sphere with these parameters.

        """
        klein = self.ideal_basis_coords()

        poincare_midpoint = self.space.kleinian_to_poincare(
            klein.sum(axis=-2) / klein.shape[-2]
        )

        poincare_extreme = utils.sphere_inversion(poincare_midpoint)

        center = (poincare_midpoint + poincare_extreme) / 2
        radius = np.sqrt(utils.normsq(poincare_midpoint - poincare_extreme)) / 2

        return center, radius

    def circle_parameters(self, short_arc=True, degrees=True,
                          coords="poincare"):
        """Get parameters describing a circular arc corresponding to this
        subspace in the Poincare model.

        Returns a tuple (center, radius, thetas). In the Poincare
        model, the subspace lies on a circle with this center and
        radius. thetas is an ndarray with shape (..., 2), giving the
        angle (with respect to the center of the circle) of two of the
        ideal points in this subspace.

        """
        if coords != "poincare":
            #TODO: don't let this pass silently (issue a warning)
            pass

        center, radius = self.sphere_parameters()
        klein = self.ideal_basis_coords()
        thetas = utils.circle_angles(center, klein)

        if short_arc:
            thetas = utils.short_arc(thetas)

        if degrees:
            thetas *= 180 / np.pi

        return center, radius, thetas

class PointPair(Point):
    def __init__(self, space, endpoint1, endpoint2=None):
        self.unit_ndims = 2
        self.space = space

        if endpoint2 is None:
            try:
                self._construct_from_object(endpoint1)
                return
            except (AttributeError, TypeError, GeometryError):
                pass

            try:
                self.set(endpoint1)
                return
            except (AttributeError, GeometryError):
                pass

        self.set_endpoints(endpoint1, endpoint2)

    def set(self, hyp_data):
        HyperbolicObject.set(self, hyp_data)
        self.endpoints = self.hyp_data[..., :2, :]

    def set_endpoints(self, endpoint1, endpoint2=None):
        """Set the endpoints of a segment.

        If endpoint2 is None, expect endpoint1 to be an array of
        points with shape (..., 2, n). Otherwise, expect endpoint1 and
        endpoint2 to be arrays of points with the same shape.

        """
        if endpoint2 is None:
            self.set(endpoint1)
            return

        pt1 = Point(self.space, endpoint1)
        pt2 = Point(self.space, endpoint2)
        self.set(np.stack([pt1.hyp_data, pt2.hyp_data], axis=-2))

    def get_endpoints(self):
        return Point(self.space, self.endpoints)

    def get_end_pair(self, as_points=False):
        """Return a pair of point objects, one for each endpoint
        """
        if as_points:
            p1, p2 = self.get_end_pair(as_points=False)
            return (Point(self.space, p1), Point(self.space, p2))
        return (self.endpoints[..., 0, :], self.endpoints[..., 1, :])

    def endpoint_coords(self, coords="kleinian"):
        return self.space.kleinian_coords(self.endpoints)

class Geodesic(PointPair, Subspace):
    def set(self, hyp_data):
        HyperbolicObject.set(self, hyp_data)
        self.endpoints = self.hyp_data[..., :2, :]
        self.ideal_basis = self.hyp_data[..., :2, :]

    def from_reflection(space, reflection):
        if space.dimension != 2:
            raise GeometryError("Creating segment from reflection expects dimension 2, got dimension {}".format(space.dimension))

        hyperplane = Hyperplane.from_reflection(space, reflection)
        pt1 = hyperplane.ideal_basis[..., 0, :]
        pt2 = hyperplane.ideal_basis[..., 1, :]
        return Geodesic(space, pt1, pt2)

class Segment(Geodesic):
    """Model a geodesic segment in hyperbolic space."""

    def set_endpoints(self, endpoint1, endpoint2=None):
        """Set the endpoints of a segment.

        If endpoint2 is None, expect endpoint1 to be an array of
        points with shape (..., 2, n). Otherwise, expect endpoint1 and
        endpoint2 to be arrays of points with the same shape.

        """
        if endpoint2 is None:
            self.endpoints = Point(self.space, endpoint1).hyp_data
            self._compute_ideal_endpoints(endpoint1)
            return

        pt1 = Point(self.space, endpoint1)
        pt2 = Point(self.space, endpoint2)
        self.endpoints = Point(self.space, np.stack(
            [pt1.hyp_data, pt2.hyp_data], axis=-2)
        )

        self._compute_ideal_endpoints(self.endpoints)

    def _assert_geometry_valid(self, hyp_data):
        HyperbolicObject._assert_geometry_valid(self, hyp_data)

        if (hyp_data.shape[-2] != 4):
            raise GeometryError( ("Underlying data for a hyperbolic"
            " segment in H{} must have shape (..., 4, {}) but data"
            " has shape {}").format(self.space.dimension,
                                   self.space.dimension + 1, hyp_data.shape))

        if not CHECK_LIGHT_CONE:
            return

        if not self.space.all_timelike(hyp_data[..., 0, :, :]):
            raise GeometryError( "segment data at index [..., 0, :,"
            ":] must consist of timelike vectors" )

        if not self.space.all_lightlike(hyp_data[..., 1, :, :]):
            raise GeometryError( "segment data at index [..., 1, :,"
            ":] must consist of lightlike vectors" )

    def set(self, hyp_data):
        HyperbolicObject.set(self, hyp_data)

        self.endpoints = self.hyp_data[..., :2, :]
        self.ideal_basis = self.hyp_data[..., 2:, :]

    def _compute_ideal_endpoints(self, endpoints):
        end_data = Point(self.space, endpoints).hyp_data
        products = end_data @ self.space.minkowski() @ end_data.swapaxes(-1, -2)
        a11 = products[..., 0, 0]
        a22 = products[..., 1, 1]
        a12 = products[..., 0, 1]

        a = a11 - 2 * a12 + a22
        b = 2 * a12 - 2 * a22
        c = a22

        mu1 = (-b + np.sqrt(b * b - 4 * a * c)) / (2*a)
        mu2 = (-b - np.sqrt(b * b - 4 * a * c)) / (2*a)

        null1 = ((mu1.T * end_data[..., 0, :].T).T +
                 ((1 - mu1).T * end_data[..., 1, :].T).T)
        null2 = ((mu2.T * end_data[..., 0, :].T).T +
                 ((1 - mu2).T * end_data[..., 1, :].T).T)

        ideal_basis = np.array([null1, null2]).swapaxes(0, -2)
        hyp_data = np.concatenate([end_data, ideal_basis], axis=-2)

        self.set(hyp_data)

    def get_ideal_endpoints(self):
        return Subspace(self.space, self.ideal_basis)

    def circle_parameters(self, short_arc=True, degrees=True,
                          coords="poincare"):
        """Get parameters describing a circular arc corresponding to this
        segment in the Poincare model.

        Returns a tuple (center, radius, thetas). In the Poincare
        model, the segment lies on a circle with this center and
        radius. thetas is an ndarray with shape (..., 2), giving the
        angle (with respect to the center of the circle) of the
        endpoints of the segment.

        """
        center, radius = self.sphere_parameters()

        klein = self.space.kleinian_coords(self.endpoints)
        poincare = self.space.kleinian_to_poincare(klein)

        thetas = utils.circle_angles(center, poincare)
        if short_arc:
            thetas = utils.short_arc(thetas)

        if degrees:
            thetas *= 180 / np.pi

        return center, radius, thetas


class Hyperplane(Subspace):
    """Model for a geodesic hyperplane in hyperbolic space."""
    def __init__(self, space, hyperplane_data):
        self.space = space
        self.unit_ndims = 2

        try:
            self._construct_from_object(hyperplane_data)
            return
        except (TypeError, GeometryError):
            pass

        try:
            self.set(hyperplane_data)
            return
        except GeometryError:
            pass

        self._compute_ideal_basis(hyperplane_data)

    def _assert_geometry_valid(self, hyp_data):
        HyperbolicObject._assert_geometry_valid(self, hyp_data)

        n = self.space.dimension + 1
        if hyp_data.shape[-2] != n:
            raise GeometryError( ("Underlying data for hyperplane in H^n must"
                                  " have shape (..., {0}, {0}) but data has shape"
                                  " {1}").format(n, hyp_data.shape))


        if not CHECK_LIGHT_CONE:
            return

        if not self.space.all_spacelike(hyp_data[..., 0, :]):
            raise GeometryError( ("Hyperplane data at index [..., 0, :]"
                                  " must consist of spacelike vectors") )

        if not self.space.all_lightlike(hyp_data[..., 1:, :]):
            raise GeometryError( ("Hyperplane data at index [..., 1, :]"
                                  " must consist of lightlike vectors") )

    def set(self, hyp_data):
        HyperbolicObject.set(self, hyp_data)

        self.spacelike_vector = self.hyp_data[..., 0, :]
        self.ideal_basis = self.hyp_data[..., 1:, :]

    def _compute_ideal_basis(self, vector):
        spacelike_vector = DualPoint(self.space, vector).hyp_data

        n = self.space.dimension + 1
        transform = self.space.spacelike_to(spacelike_vector)

        if len(spacelike_vector.shape) < 2:
            spacelike_vector = np.expand_dims(spacelike_vector, axis=0)

        standard_ideal_basis = np.vstack(
            [np.ones((1, n-1)), np.eye(n - 1, n - 1, -1)]
        )
        standard_ideal_basis[n-1, n-2] = -1.


        ideal_basis = transform.apply(standard_ideal_basis.T,
                                      broadcast="pairwise").hyp_data
        hyp_data = np.concatenate([spacelike_vector, ideal_basis],
                                  axis=-2)
        self.set(hyp_data)

    def from_reflection(space, reflection):
        """construct a hyperplane which is the fixpoint set of a given
        reflection."""
        try:
            matrix = reflection.matrix.swapaxes(-1, -2)
        except AttributeError:
            matrix = reflection

        #numpy's eig expects a matrix operating on the left
        evals, evecs = np.linalg.eig(matrix)

        #we expect a reflection to have eigenvalues [-1, 1, ...]
        expected_evals = np.ones(space.dimension + 1)
        expected_evals[0] = -1.
        eval_differences = np.sort(evals, axis=-1) - expected_evals
        if (np.abs(eval_differences) > ERROR_THRESHOLD).any():
            raise GeometryError("Not a reflection matrix")

        #sometimes eigenvalues will be complex due to roundoff error
        #so we cast to reals to avoid warnings.
        reflected = np.argmin(np.real(evals), axis=-1)

        spacelike = np.take_along_axis(
            np.real(evecs), np.expand_dims(reflected, axis=(-1,-2)), axis=-1
        )

        return Hyperplane(space, spacelike.swapaxes(-1,-2))

    def boundary_arc_parameters(self, degrees=True):
        bdry_arcs = BoundaryArc(self.space, self.ideal_basis)
        center, radius, thetas = bdry_arcs.circle_parameters(degrees=degrees)

        signs = np.sign(np.linalg.det(self.hyp_data))

        thetas[signs < 0] = np.flip(
            thetas[signs < 0], axis=-1
        )
        return center, radius, thetas

class TangentVector(HyperbolicObject):
    """Model for a tangent vector in hyperbolic space."""
    def __init__(self, space, point_data, vector=None):
        self.space = space
        self.unit_ndims = 2
        if vector is None:
            try:
                self._construct_from_object(point_data)
                return
            except TypeError:
                pass

            self.set(point_data)
            return

        self._compute_data(point_data, vector)

    def _assert_geometry_valid(self, hyp_data):
        super()._assert_geometry_valid(hyp_data)

        if hyp_data.shape[-2] != 2:
            raise GeometryError(
                ("Underlying data for a tangent vector in H{} must have shape"
                 " (..., 2, {}) but data has shape {}").format(
                     self.space.dimension, self.space.dimension + 1, hyp_data.shape)
            )

        if not CHECK_LIGHT_CONE:
            return

        point_data = hyp_data[..., 0, :]
        vector_data = hyp_data[..., 1, :]

        if not self.space.all_timelike(point_data):
            raise GeometryError("tangent vector data at index [..., 0, :] must"
                                " consist of timelike vectors")

        products = utils.apply_bilinear(point_data, vector_data, self.space.minkowski())
        if (np.abs(products) > ERROR_THRESHOLD).any():
            raise GeometryError("tangent vector must be orthogonal to point")

    def _compute_data(self, point, vector):
        #wrapping these as hyperbolic objects first
        pt = Point(self.space, point)

        #this should be a dual point, but we don't expect one until
        #after we project
        vec = HyperbolicObject(self.space, vector)

        projected = self.space.project_to_hyperboloid(pt.hyp_data, vec.hyp_data)

        self.set(np.stack([pt.hyp_data, projected], axis=-2))

    def set(self, hyp_data):
        super().set(hyp_data)
        self.point = self.hyp_data[..., 0, :]
        self.vector = self.hyp_data[..., 1, :]

    def normalized(self):
        """Get a unit tangent vector in the same direction as this tangent
        vector.

        """
        normed_vec = utils.normalize(self.vector, self.space.minkowski())
        return TangentVector(self.space, self.point, normed_vec)

    def origin_to(self, force_oriented=True):
        """Get an isometry taking the "origin" (a tangent vector pointing
        along the second standard basis vector) to this vector.

        """
        normed = utils.normalize(self.hyp_data, self.space.minkowski())
        isom = utils.find_isometry(self.space.minkowski(), normed,
                                   force_oriented)

        return Isometry(self.space, isom, column_vectors=False)

    def isometry_to(self, other, force_oriented=True):
        """Get an isometry taking this tangent vector to a scalar multiple of
        the other."""
        return other.origin_to(force_oriented) @ self.origin_to(force_oriented).inv()

    def angle(self, other):
        """Compute the angle between two tangent vectors.

        """
        v1 = self.space.project_to_hyperboloid(self.point, self.normalized().vector)
        v2 = self.space.project_to_hyperboloid(self.point, other.normalized().vector)

        product = utils.apply_bilinear(v1, v2, self.space.minkowski())
        return np.arccos(product)

    def point_along(self, distance):
        """Get a point in hyperbolic space along the geodesic specified by
        this tangent vector.

        """
        kleinian_shape = list(self.point.shape)
        kleinian_shape[-1] -= 1

        kleinian_pt = np.zeros(kleinian_shape)
        kleinian_pt[..., 0] = self.space.hyp_to_affine_dist(distance)

        basepoint = Point(self.space, kleinian_pt, coords="kleinian")

        return self.origin_to().apply(basepoint, "elementwise")

class Horosphere(HyperbolicObject):
    """Model for a horosphere in hyperbolic space

    """
    def __init__(self, space, center, reference_point=None):
        self.unit_ndims = 2
        self.space = space
        if reference_point is None:
            try:
                self._construct_from_object(center)
                return
            except TypeError:
                pass

            try:
                self.set(center)
                return
            except (AttributeError, GeometryError):
                pass

        self.set_center_ref(center, reference_point)

    def set(self, hyp_data):
        HyperbolicObject.set(self, hyp_data)
        self.center = hyp_data[..., 0, :]
        self.reference = hyp_data[..., 1, :]

    def set_center_ref(self, center, reference_point):
        centers = IdealPoint(self.space, center)
        refs = Point(self.space, reference_point)

        hyp_data = np.stack([centers.hyp_data, refs.hyp_data], axis=-2)
        self.set(hyp_data)

    def circle_parameters(self, coords="poincare"):
        ideal_coords = self.space.kleinian_coords(self.center)
        interior_coords = self.space.kleinian_coords(self.reference)

        if coords == "poincare":
            ideal_coords = self.space.kleinian_to_poincare(ideal_coords)
            interior_coords = self.space.kleinian_to_poincare(interior_coords)
        else:
            #TODO: do some kleinian horosphere computations
            pass

        model_radius = (utils.normsq(ideal_coords - interior_coords) /
                        (2 * (1 - utils.apply_bilinear(ideal_coords,
                                                       interior_coords))))

        model_center = ideal_coords * (1 - model_radius)

        return model_center, model_radius

    def intersect_geodesic(self, geodesic, p2=None):
        segment = Segment(self.space, geodesic, p2)
        geodesic_endpts = IdealPoint(self.space,
                                     segment.ideal_basis).kleinian_coords()

        # compute an isometry taking this geodesic to one which passes
        # through the origin (so we can do intersection calculations
        # in Euclidean geometry)
        midpt = Point(self.space,
                      (geodesic_endpts[..., 0, :] + geodesic_endpts[..., 1, :]) / 2.,
                       coords="kleinian")
        coord_change = midpt.origin_to()
        coord_inv = coord_change.inv()

        c_hsph = coord_inv @ self
        c_seg = coord_inv @ segment

        p_center, p_radius = c_hsph.circle_parameters(coords="poincare")
        geodesic_endpts = IdealPoint(self.space,
                                     c_seg.ideal_basis).kleinian_coords()

        u = geodesic_endpts[..., 0, :]
        v = geodesic_endpts[..., 1, :]

        a = utils.normsq(u - v)
        b = 2 * ((utils.apply_bilinear(p_center, v - u) +
                  utils.apply_bilinear(u, v) -
                  utils.normsq(u)))
        c = utils.normsq(v - p_center) - p_radius**2

        #TODO: check to see if the intersection actually occurs
        t1 = (-1 * b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        t2 = (-1 * b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)

        p1_poincare = t1 * u + (1 - t1) * v
        p2_poincare = t2 * u + (1 - t2) * v

        poincare_pts = np.stack([p1_poincare, p2_poincare], axis=-2)
        klein_pts = self.space.poincare_to_kleinian(poincare_pts)

        return coord_change @ Point(self.space, klein_pts, coords="kleinian")

class HorosphereArc(Horosphere):
    """Model for an arc lying along a horosphere

    """
    def __init__(self, space, p1, p2=None, center=None):
        self.unit_ndims = 2
        self.space = space

        if p2 is None and center is None:
            try:
                self._construct_from_object(p1)
                return
            except TypeError:
                pass
            try:
                self.set(p1)
                return
            except (AttributeError, GeometryError):
                pass

        if center is None:
            self.set_endpoints(p1, p2)
            return

        self.set_data(p1, p2, center)

    def _compute_center(self, endpoints):
        #TODO: actually compute the center, given endpoints
        pass

    def set_endpoints(self, p1, p2=None):
        if p2 is None:
            self._compute_center(p1)
            return

        endpts = np.stack(p1, p2, axis=-2)
        self._compute_center(endpts)

    def set(self, hyp_data):
        HyperbolicObject.set(self, hyp_data)
        self.center = self.hyp_data[..., 0, :]
        self.reference = self.hyp_data[..., 1, :]
        self.endpoints = self.hyp_data[..., 1:, :]


    def set_data(self, p1, p2, center):
        if p2 is None:
            pt_data = Point(self.space, p1)
            pt_1 = pt_data[..., 0, :]
            pt_2 = pt_data[..., 1, :]
        else:
            pt_1 = Point(self.space, p1)
            pt_2 = Point(self.space, p2)

        center_pt = IdealPoint(self.space, center)

        hyp_data = np.stack([center_pt.hyp_data,
                             pt_1.hyp_data,
                             pt_2.hyp_data], axis=-2)

        self.set(hyp_data)

    def circle_parameters(self, coords="poincare", degrees=True):
        center, radius = Horosphere.circle_parameters(self, coords=coords)

        model_coords = self.space.kleinian_coords(self.endpoints)
        if coords == "poincare":
            model_coords = self.space.kleinian_to_poincare(model_coords)

        thetas = utils.circle_angles(center, model_coords)

        center_theta = utils.circle_angles(
            center, self.space.kleinian_coords(self.center)
        )[..., 0]

        thetas = np.flip(utils.arc_include(thetas, center_theta), axis=-1)

        if degrees:
            thetas *= 180 / np.pi

        return center, radius, thetas

class BoundaryArc(Geodesic):
    def __init__(self, space, endpoint1, endpoint2=None):
        PointPair.__init__(self, space, endpoint1, endpoint2)

    def set(self, hyp_data):
        HyperbolicObject.set(self, hyp_data)
        self.endpoints = hyp_data[..., :2, :]
        self.orientation_pt = hyp_data[..., 2, :]

    def set_endpoints(self, endpoint1, endpoint2):
        if endpoint2 is None:
            endpoint_data = endpoint1
        else:
            pt1 = Point(self.space, endpoint1)
            pt2 = Point(self.space, endpoint2)
            endpoint_data = np.stack(
                [pt1.hyp_data, pt2.hyp_data], axis=-2
            )

        self._build_orientation_point(endpoint_data)

    def orientation(self):
        return np.linalg.det(self.hyp_data)

    def flip_orientation(self):
        self.hyp_data[..., 2, :] *= -1
        self.set(self.hyp_data)

    def _build_orientation_point(self, endpoints):
        orientation_pt_1 = np.zeros(endpoints.shape[:-2] +
                                    (1, self.space.dimension + 1))
        orientation_pt_2 = np.zeros(endpoints.shape[:-2] +
                                    (1, self.space.dimension + 1))
        orientation_pt_1[..., 0, 0] = 1.0
        orientation_pt_2[..., 0, 1] = 1.0

        point_data = np.concatenate((endpoints, orientation_pt_1), axis=-2)

        dets = np.linalg.det(point_data)
        point_data[np.abs(dets) < ERROR_THRESHOLD, 2] = orientation_pt_2

        self.set(point_data)


    def circle_parameters(self, coords="poincare", degrees=True):
        center = np.zeros(self.hyp_data.shape[:-2] + (2,))
        radius = np.ones(self.hyp_data.shape[:-2])

        klein_coords = self.space.kleinian_coords(self.endpoints)

        thetas = utils.circle_angles(center, klein_coords)

        if degrees:
            thetas *= 180 / np.pi

        orientation = self.orientation()
        thetas[orientation < 0] = np.flip(
            thetas[orientation < 0], axis=-1
        )

        return center, radius, thetas

class Isometry(HyperbolicObject):
    """Model for an isometry of hyperbolic space.

    """
    def __init__(self, space, hyp_data, column_vectors=False):
        """Constructor for Isometry.

        Underlying data is stored as row vectors, but by default the
        constructor accepts matrices acting on columns, since that's
        how my head thinks.

        """
        self.space = space
        self.unit_ndims = 2

        try:
            self._construct_from_object(hyp_data)
        except TypeError:
            if column_vectors:
                self.set(hyp_data.swapaxes(-1,-2))
            else:
                self.set(hyp_data)

    def _assert_geometry_valid(self, hyp_data):
        n = self.space.dimension + 1
        if hyp_data.shape[-2:] != (n, n):
            raise GeometryError(
                ("Isometries of H^n must be ndarray of {0} x {0}"
                 " matrices, got array with shape {1}").format(
                     n, hyp_data.shape))

    def set(self, hyp_data):
        super().set(hyp_data)
        self.matrix = hyp_data
        self.hyp_data = hyp_data

    def _apply_to_data(self, hyp_data, broadcast, unit_ndims=1):
        return utils.matrix_product(hyp_data,
                                    self.matrix,
                                    unit_ndims, self.unit_ndims,
                                    broadcast=broadcast)

    def apply(self, hyp_obj, broadcast="elementwise"):
        """Apply this isometry to another object in hyperbolic space.

        Broadcast is either "elementwise" or "pairwise", treating self
        and hyp_obj as ndarrays of isometries and hyperbolic objects,
        respectively.

        hyp_obj may be either a HyperbolicObject instance or the
        underlying data of one. In either case, this function returns
        a HyperbolicObject (of the same subclass as hyp_obj, if
        applicable).

        """
        new_obj = copy(hyp_obj)

        try:
            hyp_data = new_obj.hyp_data
            product = self._apply_to_data(new_obj.hyp_data, broadcast,
                                          new_obj.unit_ndims)
            new_obj.set(product)
            return new_obj
        except AttributeError:
            pass

        #otherwise, it's an array of vectors which we'll interpret as
        #some kind of hyperbolic object
        product = self._apply_to_data(hyp_obj, broadcast)
        return HyperbolicObject(self.space, product)

    def inv(self):
        """invert the isometry"""
        return Isometry(self.space, np.linalg.inv(self.matrix))

    def __matmul__(self, other):
        return self.apply(other)

class HyperbolicRepresentation(representation.Representation):
    """Model a representation for a finitely generated group
    representation into O(n,1).

    Really this is just a convenient way of mapping words in the
    generators to hyperbolic isometries - there's no group theory
    being done here at all.

    """
    def __init__(self, space, generator_names=[], normalization_step=-1):
        self.space = space
        representation.Representation.__init__(self, generator_names,
                                               normalization_step)

    def __getitem__(self, word):
        matrix = self._word_value(word)
        return Isometry(self.space, matrix, column_vectors=True)

    def __setitem__(self, generator, isometry):
        try:
            super().__setitem__(generator, isometry.matrix.T)
            return
        except AttributeError:
            super().__setitem__(generator, isometry)

    def normalize(self, matrix):
        return utils.indefinite_orthogonalize(self.space.minkowski(), matrix)

    def from_matrix_rep(space, rep, **kwargs):
        """Construct a hyperbolic representation from a matrix
        representation"""
        hyp_rep = HyperbolicRepresentation(space, **kwargs)
        for g, matrix in rep.generators.items():
            hyp_rep[g] = matrix

        return hyp_rep

    def isometries(self, words):
        """Get an Isometry object holding the matrices which are the images of
        a sequence of words in the generators.

        """
        matrix_array = np.array(
            [representation.Representation.__getitem__(self, word)
             for word in words]
        )
        return Isometry(self.space, matrix_array, column_vectors=True)

class HyperbolicSpace:
    """Class to generate objects in hyperbolic geometry and perform some
    utility computations."""
    def __init__(self, dimension):
        self.projective = projective.ProjectiveSpace(dimension + 1)
        self.dimension = dimension

    def minkowski(self):
        """Get the matrix for the Minkowski bilinear form, with signature (-1,
        1, 1, ...., 1).

        """
        #TODO: allow for a different underlying bilinear form

        return np.diag(np.concatenate(([-1.0], np.ones(self.dimension))))

    def hyperboloid_coords(self, points):
        """Project an ndarray of points to the unit hyperboloid defined by the
        Minkowski quadratic form."""

        #last index of array as the dimension
        n = self.dimension + 1
        projective_coords, dim_index = utils.dimension_to_axis(points, n, -1)

        hyperbolized = utils.normalize(projective_coords, self.minkowski())

        return hyperbolized.swapaxes(-1, dim_index)

    def kleinian_coords(self, points):
        """Get kleinian coordinates for an ndarray of points.

        """
        n = self.dimension + 1
        projective_coords, dim_index = utils.dimension_to_axis(points, n, -1)

        return self.projective.affine_coordinates(points).swapaxes(-1, dim_index)

    def kleinian_to_poincare(self, points):
        euc_norms = utils.normsq(points)
        #we take absolute value to combat roundoff error
        mult_factor = 1 / (1. + np.sqrt(np.abs(1 - euc_norms)))

        return (points.T * mult_factor.T).T

    def poincare_to_kleinian(self, points):
        euc_norms = utils.normsq(points)
        mult_factor = 2. / (1. + euc_norms)

        return (points.T * mult_factor.T).T

    def get_elliptic(self, block_elliptic):
        """Get an elliptic isometry stabilizing the origin in the
        Poincare/Klein models.

        block_elliptic is an element of O(n), whose image is taken
        diagonally in O(n,1).

        """
        mat = scipy.linalg.block_diag(1.0, block_elliptic)

        return Isometry(self, mat, column_vectors=True)

    def project_to_hyperboloid(self, basepoint, tangent_vector):
        """Project a vector in R^(n,1) to lie in the tangent space to the unit
        hyperboloid at a given basepoint.

        """
        return tangent_vector - utils.projection(
            tangent_vector, basepoint, self.minkowski())

    def get_origin(self, shape=()):
        """Get a Point mapping to the origin of the Poincare/Klein models.

        """
        return self.get_point(np.zeros(shape + (self.dimension,)))

    def get_base_tangent(self, shape=()):
        """Get a TangentVector whose basepoint maps to the origin of the
        Poincare/Klein models, and whose vector is the second standard
        basis vector in R^(n,1).

        """
        origin = self.get_origin(shape)
        vector = np.zeros(() + (self.dimension + 1,))
        vector[..., 1] = 1

        return TangentVector(self, origin, vector)

    def get_point(self, point, coords="kleinian"):
        return Point(self, point, coords)

    def hyp_to_affine_dist(self, r):
        """Convert distance in hyperbolic space to distance from the origin in
        the Klein model.

        """
        return (np.exp(2 * r) - 1) / (1 + np.exp(2 * r))

    def _loxodromic_basis_change(self):
        basis_change = np.array([
            [1.0, 1.0],
            [1.0, -1.0]
        ])
        return scipy.linalg.block_diag(
            basis_change, np.identity(self.dimension - 1)
        )

    def get_standard_loxodromic(self, parameter):
        """Get a loxodromic isometry whose axis intersects the origin.

        WARNING: not vectorized.

        """

        basis_change = self._loxodromic_basis_change()
        diagonal_loxodromic = np.diag(
            np.concatenate(([parameter, 1.0/parameter],
                       np.ones(self.dimension - 1)))
        )

        return Isometry(self,
                        basis_change @ diagonal_loxodromic @ np.linalg.inv(basis_change),
                        column_vectors=True
        )

    def get_standard_rotation(self, angle):
        """Get a rotation about the origin by a fixed angle.

        WARNING: not vectorized.
        """
        affine = scipy.linalg.block_diag(
            utils.rotation_matrix(angle),
            np.identity(self.dimension - 2)
        )
        return self.get_elliptic(affine)

    def regular_polygon(self, n, hyp_radius):
        """Get a regular polygon with n vertices, inscribed on a circle of
        radius hyp_radius.

        (This is actually vectorized.)

        """
        radius = np.array(hyp_radius)
        tangent = self.get_base_tangent(radius.shape).normalized()
        start_vertex = tangent.point_along(radius)

        cyclic_rep = HyperbolicRepresentation(self)
        cyclic_rep["a"] = self.get_standard_rotation(2 * np.pi / n)

        words = ["a" * i for i in range(n)]
        mats = cyclic_rep.isometries(words)

        vertices = mats.apply(start_vertex, "pairwise_reversed")
        return vertices

    def regular_polygon_radius(self, n, interior_angle):
        """Find r such that a regular n-gon inscribed on a circle of radius r
        has the given interior angle.

        """
        alpha = interior_angle / 2
        gamma = np.pi / n
        term = ((np.cos(alpha)**2 - np.sin(gamma)**2) /
                ((np.sin(alpha) * np.sin(gamma))**2))
        return np.arcsinh(np.sqrt(term))

    def polygon_interior_angle(self, n, hyp_radius):
        """Get the interior angle of a regular n-gon inscribed on a circle
        with the given hyperbolic radius.

        """
        gamma = np.pi / n
        denom = np.sqrt(1 + (np.sin(gamma) * np.sinh(hyp_radius))**2)
        return 2 * np.arcsin(np.cos(gamma) / denom)

    def genus_g_surface_radius(self, g):
        """Find the radius of a regular polygon giving the fundamental domain
        for the action of a hyperbolic surface group with genus g.

        """
        return self.regular_polygon_radius(4 * g, np.pi / (4*g))

    def regular_surface_polygon(self, g):
        """Get a regular polygon which is the fundamental domain for the
        action of a hyperbolic surface group with genus g.

        """
        return self.regular_polygon(4 * g, self.genus_g_surface_radius(g))



    def close_to_boundary(self, vectors):
        """Return true if all of the given vectors have kleinian coords close
        to the boundary of H^n.

        """
        return (np.abs(utils.normsq(self.kleinian_coords(vectors)) - 1)
                < ERROR_THRESHOLD).all()

    def all_spacelike(self, vectors):
        """Return true if all of the vectors are spacelike.

        Highly susceptible to round-off error, probably don't rely on
        this.

        """
        return (utils.normsq(vectors, self.minkowski()) > ERROR_THRESHOLD).all()

    def all_timelike(self, vectors):
        """Return true if all of the vectors are timelike.

        Highly susceptible to round-off error, probably don't rely on
        this.

        """
        return (utils.normsq(vectors, self.minkowski()) < ERROR_THRESHOLD).all()

    def all_lightlike(self, vectors):
        """Return true if all of the vectors are timelike.

        Highly susceptible to round-off error, probably don't rely on
        this.

        """
        return (np.abs(utils.normsq(vectors, self.minkowski())) < ERROR_THRESHOLD).all()


    def timelike_to(self, v, force_oriented=False):
        """Find an isometry taking the origin of the Poincare/Klein models to
        the given vector v.

        We expect v to be timelike in order for this to make sense.

        """
        if not self.all_timelike(v):
            raise GeometryError( "Cannot find isometry taking a"
            " timelike vector to a non-timelike vector."  )

        return Isometry(self, utils.find_isometry(self.minkowski(),
                                                  v, force_oriented),
                        column_vectors=False)

    def spacelike_to(self, v, force_oriented=False):
        """Find an isometry taking the second standard basis vector (0, 1, 0,
        ...) to the given vector v.

        We expect v to be spacelike in order for this to make sense.


        """
        normed = utils.normalize(v, self.minkowski())

        if not self.all_spacelike(v):
            raise GeometryError( "Cannot find isometry taking a"
            " spacelike vector to a non-spacelike vector.")

        iso = utils.find_isometry(self.minkowski(), normed)

        #find the index of the timelike basis vector
        lengths = np.expand_dims(utils.normsq(iso, self.minkowski()), axis=-1)
        t_index = np.argmin(lengths, axis=-2)

        #apply a permutation so the isometry actually preserves the
        #form. we do the permutation in two steps because it could be
        #either a 2-cycle or a 3-cycle.
        p_iso = iso.copy()

        #first swap timelike index with zero
        indices = np.stack([np.zeros_like(t_index), t_index], axis=-2)
        p_indices = np.stack([t_index, np.zeros_like(t_index)], axis=-2)

        p_values = np.take_along_axis(iso, indices, axis=-2)
        np.put_along_axis(p_iso, p_indices, p_values, axis=-2)

        #then swap timelike index with one
        indices = np.stack([np.ones_like(t_index), t_index], axis=-2)
        p_indices = np.stack([t_index, np.ones_like(t_index)], axis=-2)

        p_values = np.take_along_axis(p_iso, indices, axis=-2)
        np.put_along_axis(p_iso, p_indices, p_values, axis=-2)

        if force_oriented:
            p_iso = utils.make_orientation_preserving(p_iso)

        return Isometry(self, p_iso, column_vectors=False)

class HyperbolicPlane(HyperbolicSpace):
    """Hyperbolic space of dimension 2.

    None of these functions are properly vectorized, so don't use them
    (none of them are all that useful anyway)

    """
    def __init__(self):
        self.projective = projective.ProjectivePlane()
        self.dimension = 2

    def get_boundary_point(self, theta):
        return IdealPoint(self, np.array([1.0, np.cos(theta), np.sin(theta)]))

    def basis_change(self):
        basis_change = np.array([
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        return basis_change

    def get_standard_reflection(self):
        return Isometry(self, np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0]]),
                        column_vectors=True)
