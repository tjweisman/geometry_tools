"""Model objects in hyperbolic space with numerical coordinates.

"""

from copy import copy
from enum import Enum

import numpy as np

from geometry_tools import projective, representation, utils
from geometry_tools.projective import GeometryError

#arbitrary
ERROR_THRESHOLD = 1e-8
BOUNDARY_THRESHOLD = 1e-5

CHECK_LIGHT_CONE = False


class Model(Enum):
    """Enumerate implemented models of hyperbolic space.

    Models can have different aliases, and can be compared to strings
    with the == operator, which returns `True` if the strings match
    any alias name (case insensitive).

    """
    POINCARE = "poincare"
    KLEIN = "klein"
    KLEINIAN = "klein"
    AFFINE = "klein"
    HALFSPACE = "halfspace"
    HALFPLANE = "halfspace"
    HYPERBOLOID = "hyperboloid"
    PROJECTIVE = "projective"

    def aliases(self):
        """List all of the different accepted names for this hyperbolic model."""
        return [name for name, member in Model.__members__.items()
                if member is self]

    def __eq__(self, other):
        if self is other:
            return True

        try:
            if other.upper() in self.aliases():
                return True
        except AttributeError:
            pass

        return False

class HyperbolicObject(projective.ProjectiveObject):
    """Model for an abstract object in hyperbolic space.

    Subclasses of HyperbolicObject model more specific objects in
    hyperbolic geometry, e.g. points, geodesic segments, hyperplanes,
    ideal points.

    Every object in hyperbolic space H^n has the underlying data of an
    ndarray whose last dimension is n+1. This is interpreted as a
    collection of (row) vectors in R^(n+1), or more precisely R^(n,1),
    which transforms via the action of the indefinite orthogonal group
    O(n,1).

    The last `unit_ndims` dimensions of this underlying array represent
    a single object in hyperbolic space, which transforms as a unit
    under the action of O(n,1). The remaining dimensions of the array
    are used to represent an array of such "unit objects."

    """

    @property
    def minkowski(self):
        return minkowski(self.dimension + 1)


    def coords(self, model, proj_data=None):
        """Get or set a representation of this hyperbolic object in
        coordinates.

        For the base `HyperbolicObject` class, the available models
        are `Model.KLEIN` and `Model.PROJECTIVE`.

        Parameters
        ----------
        model : Model
            which model to take coordinates in
        proj_data : ndarray
            data for this hyperbolic object, interpreted as
            coordinates with respect to `model`. If `None`, just
            return coordinates.

        Raises
        ------
        GeometryError
            Raised if an unsupported model is specified.

        """
        if model == Model.KLEIN:
            return self.kleinian_coords(proj_data)
        if model == Model.PROJECTIVE:
            return self.projective_coords(proj_data)

        raise GeometryError(
            "Unimplemented model for an object of type {}: '{}'".format(
                self.__class__.__name__, model
            ))

    def kleinian_coords(self, aff_data=None):
        return self.affine_coords(aff_data, chart_index=0)

class Point(HyperbolicObject, projective.Point):
    """Model for a point (or ndarray of points) in the closure of
    hyperbolic space.

    """
    def __init__(self, point, model=Model.PROJECTIVE):
        self.unit_ndims = 1
        self.aux_ndims = 0

        try:
            self._construct_from_object(point)
            return
        except TypeError:
            pass

        self.coords(model, point)

    def hyperboloid_coords(self, proj_data=None):
        """Get or set point coordinates on the unit hyperboloid.
        """
        if proj_data is not None:
            self.set(proj_data)

        return hyperboloid_coords(self.proj_data)

    def poincare_coords(self, proj_data=None):
        """Get or set point coordinates in the Poincare ball (radius 1)
        """
        if proj_data is not None:
            klein = poincare_to_kleinian(np.array(proj_data))
            self.kleinian_coords(klein)

        return kleinian_to_poincare(self.kleinian_coords())

    def halfspace_coords(self, proj_data=None):
        """Get or set point coordinates in the upper half-space model

        Parameters
        ----------
        proj_data : ndarray

        """

        poincare = None

        if proj_data is not None:
            poincare = halfspace_to_poincare(np.array(proj_data))

        poincare_coords = self.poincare_coords(poincare)
        return poincare_to_halfspace(poincare_coords)

    def coords(self, model, proj_data=None):
        """Get or set a representation of this hyperbolic object in
        coordinates.

        The available models are `Model.KLEIN`, `Model.PROJECTIVE`,
        `Model.POINCARE`, `Model.HYPERBOLOID`, and `Model.HALFSPACE`.

        Parameters
        ----------
        model : Model
            which model to take coordinates in
        proj_data : ndarray
            data for this hyperbolic object, interpreted as
            coordinates with respect to `model`. If `None`, just
            return coordinates.

        Raises
        ------
        GeometryError
            Raised if an unsupported model is specified.

        """
        try:
            return HyperbolicObject.coords(self, model, proj_data)
        except GeometryError as e:
            if model == Model.POINCARE:
                return self.poincare_coords(proj_data)
            if model == Model.HYPERBOLOID:
                return self.hyperboloid_coords(proj_data)
            if model == Model.HALFSPACE:
                return self.halfspace_coords(proj_data)
            raise e

    def distance(self, other):
        """compute elementwise distances (with respect to last two dimensions)
        between `self` and `other`

        """
        #TODO: allow for pairwise distances
        products = utils.apply_bilinear(self.proj_data, other.proj_data,
                                        self.minkowski)

        return np.arccosh(np.abs(products))

    def origin_to(self):
        """Get an isometry taking the origin to `self`.

        Not guaranteed to be orientation-preserving."""
        return timelike_to(self.proj_data)

    def unit_tangent_towards(self, other):
        """Get the unit tangent vector pointing from `self` to `other`.

        """
        diff = other.proj_data - self.proj_data
        return TangentVector(self, diff).normalized()

    def get_origin(dimension, shape=()):
        """Get a Point mapping to the origin of the Poincare/Klein models.

        """
        return Point(np.zeros(shape + (dimension,)), model="klein")

class DualPoint(Point):
    """Model for a "dual point" in hyperbolic space (a point in the
    complement of the projectivization of the Minkowski light cone,
    corresponding to a geodesic hyperplane in hyperbolic space).

    """

    def _assert_geometry_valid(self, proj_data):
        Point._assert_geometry_valid(self, proj_data)
        if not CHECK_LIGHT_CONE:
            return

        if not all_spacelike(proj_data):
            raise GeometryError("Dual point data must consist of vectors"
                                " in the complement of the Minkowski light cone")

class IdealPoint(Point):
    """Model for an ideal point in hyperbolic space (lying on the boundary
    of the projectivization of the Minkowski light cone)

    """
    def _assert_geometry_valid(self, proj_data):
        super()._assert_geometry_valid(proj_data)

        if not CHECK_LIGHT_CONE:
            return

        if not all_lightlike(proj_data):
            raise GeometryError("Ideal point data must consist of vectors"
                                "in the boundary of the Minkowski light cone")

    def from_angle(theta):
        #TODO: vectorize and make this work in n dimensions
        return IdealPoint(np.array([1.0, np.cos(theta), np.sin(theta)]))

class Subspace(IdealPoint):
    """Model for a geodesic subspace of hyperbolic space.

    """
    def __init__(self, proj_data):
        HyperbolicObject.__init__(self, proj_data, unit_ndims=2)

    def set(self, proj_data):
        HyperbolicObject.set(self, proj_data)
        self.ideal_basis = proj_data

    def ideal_basis_coords(self, model=Model.KLEIN):
        """Get coordinates for a basis of this subspace lying on the ideal
        boundary of H^n.

        """
        #we don't return proj_data directly, since we want subclasses
        #to be able to use this method even if the structure of the
        #underlying data is different.
        return Point(self.ideal_basis).coords(model)

    def spacelike_complement(self):
        orthed = self._data_with_dual()
        return DualPoint(orthed[..., 0, :])

    def _data_with_dual(self):
        midpoints = np.sum(self.ideal_basis, axis=-2) / self.ideal_basis.shape[-2]

        spacelike_guess = np.ones(
            self.ideal_basis.shape[:-2] + (1, self.ideal_basis.shape[-1])
        )
        spacelike_guess[..., 0] = 0.

        to_orthogonalize = np.concatenate([
            np.expand_dims(midpoints, axis=-2),
            self.ideal_basis[..., 1:, :],
            spacelike_guess], axis=-2)

        orthed = utils.indefinite_orthogonalize(self.minkowski,
                                                to_orthogonalize)
        return np.concatenate([
            np.expand_dims(orthed[..., -1, :], axis=-2),
            self.ideal_basis], axis=-2)

    def sphere_parameters(self, model=Model.POINCARE):
        """Get parameters describing a k-sphere corresponding to this subspace
        in the Poincare model.

        Parameters
        ----------
        model : Model
            hyperbolic model to use. Acceptable values are
            `Model.POINCARE` and `Model.HALFSPACE`

        Returns
        ---------
        tuple
            Tuple of the form `(centers, radii)`, where `centers` and
            `radii` are `ndarray`s respectively holding the centers
            and radii of spheres corresponding to this subspace in the
            given hyperbolic model.

        Raises
        ------
        GeometryError

    """

        if model == Model.POINCARE:
            klein_basis = self.ideal_basis_coords(model=Model.KLEIN)
            klein_midpoint = klein_basis.sum(axis=-2) / klein_basis.shape[-2]
            poincare_midpoint = kleinian_to_poincare(klein_midpoint)
            poincare_extreme = utils.sphere_inversion(poincare_midpoint)

            center = (poincare_midpoint + poincare_extreme) / 2
            radius = np.sqrt(utils.normsq(poincare_midpoint - poincare_extreme)) / 2

        elif model == Model.HALFSPACE:
            halfspace_basis = self.ideal_basis_coords(model=Model.HALFSPACE)
            halfspace_midpoint = (halfspace_basis.sum(axis=-2) /
                                  halfspace_basis.shape[-2])

            #just use the first element of the basis
            center = halfspace_midpoint
            radius = np.sqrt(
                utils.normsq(halfspace_basis[..., 0, :] - halfspace_midpoint)
            )
        else:
            raise GeometryError(
                ("Cannot compute spherical parameters for an object of type {}"
                 " in model: '{}'").format(self.__class__.__name__, model)
            )

        return center, radius

    def circle_parameters(self, degrees=True, model=Model.POINCARE):
        """Get parameters describing a circular arc corresponding to this
        subspace in the Poincare or halfspace models.

        Parameters
        ----------
        degrees : bool
            if `True`, return angles in degrees. Otherwise, return
            angles in radians
        model : Model
            hyperbolic model to use for the computation

        Returns
        --------
        tuple
            tuple `(centers, radii, thetas)`, where `centers`,
            `radii`, and `thetas` are `ndarray`s representing the
            centers, radii, and begin/end angles for the circle
            corresponding to this arc in the given model of hyperbolic
            space. Angles are always specified in counterclockwise
            order.

        """
        center, radius = self.sphere_parameters(model)
        endpoints = self.ideal_basis_coords(model)
        thetas = utils.circle_angles(center, endpoints)

        if model == Model.POINCARE:
            thetas = utils.short_arc(thetas)

        if model == Model.HALFSPACE:
            thetas = utils.right_to_left(thetas)

        if degrees:
            thetas *= 180 / np.pi

        return center, radius, thetas

    def reflection_across(self):
        """Get a hyperbolic isometry reflecting across this hyperplane.

        Returns
        --------
        Isometry
            Isometry reflecting across this hyperplane.

        """
        dual_data = self._data_with_dual()
        if self.dimension + 1 != dual_data.shape[-2]:
            raise GeometryError(
                ("Cannot compute a reflection across a subspace of "
                 "dimension {}").format(dual_data.shape[-2])
            )

        refdata = (np.linalg.inv(dual_data) @
                   self.minkowski @
                   dual_data)

        return Isometry(refdata, column_vectors=False)

class PointPair(Point, projective.PointPair):
    """Abstract model for a hyperbolic object whose underlying data is
    determined by a pair of points in R^(n,1)

    """
    def __init__(self, endpoint1, endpoint2=None):
        projective.PointPair.__init__(self, endpoint1, endpoint2)

    def endpoint_coords(self, model=Model.KLEIN):
        return self.get_endpoints().coords(model)

    def get_endpoints(self):
        return Point(self.endpoints)

    def get_end_pair(self, as_points=False):
        """Return a pair of point objects, one for each endpoint
        """
        if as_points:
            p1, p2 = self.get_end_pair(as_points=False)
            return (Point(p1), Point(p2))

        return (self.endpoints[..., 0, :], self.endpoints[..., 1, :])

class Geodesic(PointPair, Subspace):
    """Model for a bi-infinite gedoesic in hyperbolic space.

    """
    def set(self, proj_data):
        HyperbolicObject.set(self, proj_data)
        self.endpoints = self.proj_data[..., :2, :]
        self.ideal_basis = self.proj_data[..., :2, :]

    def from_reflection(reflection):
        if reflection.dimension != 2:
            raise GeometryError("Creating segment from reflection expects dimension 2, got dimension {}".format(reflection.dimension))

        hyperplane = Hyperplane.from_reflection(reflection)
        pt1 = hyperplane.ideal_basis[..., 0, :]
        pt2 = hyperplane.ideal_basis[..., 1, :]
        return Geodesic(pt1, pt2)

class Segment(Geodesic):
    """Model a geodesic segment in hyperbolic space."""

    def __init__(self, endpoint1, endpoint2=None):
        self.unit_ndims = 2
        self.aux_ndims = 0

        if endpoint2 is None:
            try:
                self._construct_from_object(endpoint1)
                return
            except (AttributeError, TypeError, GeometryError):
                pass

            try:
                self.set(endpoint1)
                return
            except (AttributeError, GeometryError) as e:
                pass

        self.set_endpoints(endpoint1, endpoint2)

    def set(self, proj_data):
        HyperbolicObject.set(self, proj_data)
        self.endpoints = self.proj_data[..., :2, :]
        self.ideal_basis = self.proj_data[..., 2:, :]

    def set_endpoints(self, endpoint1, endpoint2=None):
        """Set the endpoints of a segment.

        If endpoint2 is None, expect endpoint1 to be an array of
        points with shape (..., 2, n). Otherwise, expect endpoint1 and
        endpoint2 to be arrays of points with the same shape.

        """
        if endpoint2 is None:
            self.endpoints = Point(endpoint1).proj_data
            self._compute_ideal_endpoints(endpoint1)
            return

        pt1 = Point(endpoint1)
        pt2 = Point(endpoint2)
        self.endpoints = Point(np.stack(
            [pt1.proj_data, pt2.proj_data], axis=-2)
        )

        self._compute_ideal_endpoints(self.endpoints)

    def _assert_geometry_valid(self, proj_data):
        HyperbolicObject._assert_geometry_valid(self, proj_data)

        if (proj_data.shape[-2] != 4):
            raise GeometryError( ("Underlying data for a hyperbolic"
            " segment must have shape (..., 4, n) but data"
            " has shape {}").format(proj_data.shape))

        if not CHECK_LIGHT_CONE:
            return

        if not all_timelike(proj_data[..., 0, :, :]):
            raise GeometryError( "segment data at index [..., 0, :,"
            ":] must consist of timelike vectors" )

        if not all_lightlike(proj_data[..., 1, :, :]):
            raise GeometryError( "segment data at index [..., 1, :,"
            ":] must consist of lightlike vectors" )

    def _compute_ideal_endpoints(self, endpoints):
        end_data = Point(endpoints).proj_data
        dim = end_data.shape[-1]
        products = end_data @ minkowski(dim) @ end_data.swapaxes(-1, -2)
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
        proj_data = np.concatenate([end_data, ideal_basis], axis=-2)

        self.set(proj_data)

    def get_ideal_endpoints(self):
        return Geodesic(self.ideal_basis)

    def ideal_endpoint_coords(self, model=Model.KLEIN):
        """Alias for Subspace.ideal_basis_coords.

        """
        return self.ideal_basis_coords(model)

    def circle_parameters(self, degrees=True, model=Model.POINCARE):
        """Get parameters describing a circular arc corresponding to this
        segment in the Poincare or halfspace models.

        Returns a tuple `(center, radius, thetas)`. In the Poincare
        model, the segment lies on a circle with this center and
        radius. thetas is an ndarray with shape `(..., 2)`, giving the
        angle (with respect to the center of the circle) of the
        endpoints of the segment.

        """
        center, radius = self.sphere_parameters(model)

        endpoints = self.endpoint_coords(model)

        thetas = utils.circle_angles(center, endpoints)

        if model == Model.POINCARE:
            thetas = utils.short_arc(thetas)
        elif model == Model.HALFSPACE:
            thetas = utils.right_to_left(thetas)

        if degrees:
            thetas *= 180 / np.pi

        return center, radius, thetas


class Hyperplane(Subspace):
    """Model for a geodesic hyperplane in hyperbolic space."""

    #TODO: reimplement so ideal_basis is aux_data
    def __init__(self, hyperplane_data):
        self.unit_ndims = 2
        self.aux_ndims = 0

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

    def _assert_geometry_valid(self, proj_data):
        HyperbolicObject._assert_geometry_valid(self, proj_data)

        if (len(proj_data.shape) < 2 or
            proj_data.shape[-2] != proj_data.shape[-1]):
            raise GeometryError( ("Underlying data for hyperplane in H^n must"
                                  " have shape (..., n, n) but data has shape"
                                  " {}").format(proj_data.shape))


        if not CHECK_LIGHT_CONE:
            return

        if not all_spacelike(proj_data[..., 0, :]):
            raise GeometryError( ("Hyperplane data at index [..., 0, :]"
                                  " must consist of spacelike vectors") )

        if not all_lightlike(proj_data[..., 1:, :]):
            raise GeometryError( ("Hyperplane data at index [..., 1, :]"
                                  " must consist of lightlike vectors") )

    def _data_with_dual(self):
        return self.proj_data

    def set(self, proj_data):
        HyperbolicObject.set(self, proj_data)

        self.spacelike_vector = self.proj_data[..., 0, :]
        self.ideal_basis = self.proj_data[..., 1:, :]

    def _compute_ideal_basis(self, vector):
        spacelike_vector = DualPoint(vector).proj_data
        n = spacelike_vector.shape[-1]
        transform = spacelike_to(spacelike_vector)

        if len(spacelike_vector.shape) < 2:
            spacelike_vector = np.expand_dims(spacelike_vector, axis=0)

        standard_ideal_basis = np.vstack(
            [np.ones((1, n-1)), np.eye(n - 1, n - 1, -1)]
        )
        standard_ideal_basis[n-1, n-2] = -1.


        ideal_basis = transform.apply(standard_ideal_basis.T,
                                      broadcast="pairwise").proj_data
        proj_data = np.concatenate([spacelike_vector, ideal_basis],
                                  axis=-2)
        self.set(proj_data)

    def from_reflection(reflection):
        """construct a hyperplane which is the fixpoint set of a given
        reflection."""
        try:
            matrix = reflection.matrix.swapaxes(-1, -2)
        except AttributeError:
            matrix = reflection

        #numpy's eig expects a matrix operating on the left
        evals, evecs = np.linalg.eig(matrix)

        dimension = reflection.dimension

        #we expect a reflection to have eigenvalues [-1, 1, ...]
        expected_evals = np.ones(dimension + 1)
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

        return Hyperplane(spacelike.swapaxes(-1,-2))

    def boundary_arc_parameters(self, degrees=True):
        bdry_arcs = BoundaryArc(self.ideal_basis)
        center, radius, thetas = bdry_arcs.circle_parameters(degrees=degrees)

        signs = np.sign(np.linalg.det(self.proj_data))

        thetas[signs < 0] = np.flip(
            thetas[signs < 0], axis=-1
        )
        return center, radius, thetas

class TangentVector(HyperbolicObject):
    """Model for a tangent vector in hyperbolic space."""
    def __init__(self, point_data, vector=None):
        self.unit_ndims = 2
        self.aux_ndims = 0

        if vector is None:
            try:
                self._construct_from_object(point_data)
                return
            except TypeError:
                pass

            self.set(point_data)
            return

        self._compute_data(point_data, vector)

    def _assert_geometry_valid(self, proj_data):
        super()._assert_geometry_valid(proj_data)

        if proj_data.shape[-2] != 2:
            raise GeometryError(
                ("Underlying data for a tangent vector in Hn must have shape"
                 " (..., 2, n) but data has shape {}").format(
                     proj_data.shape)
            )

        if not CHECK_LIGHT_CONE:
            return

        point_data = proj_data[..., 0, :]
        vector_data = proj_data[..., 1, :]

        if not all_timelike(point_data):
            raise GeometryError("tangent vector data at index [..., 0, :] must"
                                " consist of timelike vectors")

        products = utils.apply_bilinear(point_data, vector_data, self.minkowski)
        if (np.abs(products) > ERROR_THRESHOLD).any():
            raise GeometryError("tangent vector must be orthogonal to point")

    def _compute_data(self, point, vector):
        #wrapping these as hyperbolic objects first
        pt = Point(point)

        #this should be a dual point, but we don't expect one until
        #after we project
        vec = HyperbolicObject(vector)

        projected = project_to_hyperboloid(pt.proj_data, vec.proj_data)

        self.set(np.stack([pt.proj_data, projected], axis=-2))

    def set(self, proj_data):
        HyperbolicObject.set(self, proj_data)
        self.point = self.proj_data[..., 0, :]
        self.vector = self.proj_data[..., 1, :]

    def normalized(self):
        """Get a unit tangent vector in the same direction as this tangent
        vector.

        """
        normed_vec = utils.normalize(self.vector, self.minkowski)
        return TangentVector(self.point, normed_vec)

    def origin_to(self, force_oriented=True):
        """Get an isometry taking the "origin" (a tangent vector pointing
        along the second standard basis vector) to this vector.

        """
        normed = utils.normalize(self.proj_data, self.minkowski)
        isom = utils.find_isometry(self.minkowski, normed,
                                   force_oriented)

        return Isometry(isom, column_vectors=False)

    def isometry_to(self, other, force_oriented=True):
        """Get an isometry taking this tangent vector to a scalar multiple of
        the other."""
        return other.origin_to(force_oriented) @ self.origin_to(force_oriented).inv()

    def angle(self, other):
        """Compute the angle between two tangent vectors.

        """
        v1 = project_to_hyperboloid(self.point, self.normalized().vector)
        v2 = project_to_hyperboloid(self.point, other.normalized().vector)

        product = utils.apply_bilinear(v1, v2, self.minkowski)
        return np.arccos(product)

    def point_along(self, distance):
        """Get a point in hyperbolic space along the geodesic specified by
        this tangent vector.

        """
        kleinian_shape = list(self.point.shape)
        kleinian_shape[-1] -= 1

        kleinian_pt = np.zeros(kleinian_shape)
        kleinian_pt[..., 0] = hyp_to_affine_dist(distance)

        basepoint = Point(kleinian_pt, model=Model.KLEIN)

        return self.origin_to().apply(basepoint, "elementwise")

    def get_base_tangent(dimension, shape=()):
        """Get a TangentVector whose basepoint maps to the origin of the
        Poincare/Klein models, and whose vector is the second standard
        basis vector in R^(n,1).

        """
        origin = Point.get_origin(dimension, shape)
        vector = np.zeros(() + (dimension + 1,))
        vector[..., 1] = 1.

        return TangentVector(origin, vector)

class Horosphere(HyperbolicObject):
    """Model for a horosphere in hyperbolic space

    """
    def __init__(self, center, reference_point=None):
        self.unit_ndims = 2
        self.aux_ndims = 0

        if reference_point is None:
            try:
                self._construct_from_object(center)
                return
            except TypeError:
                pass

        self.set_center_ref(center, reference_point)

    def set(self, proj_data):
        HyperbolicObject.set(self, proj_data)
        self.center = proj_data[..., 0, :]
        self.reference = proj_data[..., 1, :]

    def set_center_ref(self, center, reference_point=None):
        if reference_point is None:
            proj_data = Point(center).proj_data
        else:
            center = IdealPoint(center)
            ref = Point(reference_point)
            proj_data = np.stack([center.proj_data, ref.proj_data], axis=-2)

        self.set(proj_data)

    def sphere_parameters(self, model=Model.POINCARE):
        """Get the center and radius of a sphere giving this horosphere in
        poincare coordinates.

        """
        ideal_coords = Point(self.center).coords(model)
        ref_coords = Point(self.reference).coords(model)

        if model == Model.POINCARE:
            model_radius = (utils.normsq(ideal_coords - ref_coords) /
                            (2 * (1 - utils.apply_bilinear(ideal_coords,
                                                           ref_coords))))
            model_center = ideal_coords * (1 - model_radius[..., np.newaxis])
        elif model == Model.HALFSPACE:
            with np.errstate(divide="ignore", invalid="ignore"):
                ideal_euc = ideal_coords[..., :-1]
                ref_euc = ref_coords[..., :-1]
                ref_z = ref_coords[..., -1]

                model_radius = 0.5 * (utils.normsq(ideal_euc - ref_euc) / ref_z + ref_z)
                model_center = ideal_coords[:]
                model_center[..., -1] = model_radius
        else:
            raise GeometryError(
                "No implementation for spherical parameters for a horosphere in"
                " model: '{}'".format(model)
            )

        return model_center, model_radius

    def center_coords(self, model=Model.KLEIN):
        return Point(self.center).coords(model)

    def ref_coords(self, model):
        return Point(self.reference).coords(model)

    def intersect_geodesic(self, geodesic, p2=None):
        """Compute the intersection points of a geodesic with this horosphere

        """
        segment = Segment(geodesic, p2)
        geodesic_endpts = segment.ideal_endpoint_coords(Model.KLEIN)

        # compute an isometry taking this geodesic to one which passes
        # through the origin (so we can do intersection calculations
        # in Euclidean geometry)
        midpt = Point((geodesic_endpts[..., 0, :] + geodesic_endpts[..., 1, :]) / 2.,
                       model=Model.KLEIN)
        coord_change = midpt.origin_to()
        coord_inv = coord_change.inv()

        c_hsph = coord_inv @ self
        c_seg = coord_inv @ segment

        p_center, p_radius = c_hsph.sphere_parameters()
        geodesic_endpts = c_seg.ideal_endpoint_coords(Model.KLEIN)

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
        klein_pts = poincare_to_kleinian(poincare_pts)

        return coord_change @ Point(klein_pts, model=Model.KLEIN)

class HorosphereArc(Horosphere, PointPair):
    """Model for an arc lying along a horosphere

    """
    def __init__(self, center, p1=None, p2=None):
        self.unit_ndims = 2
        self.aux_ndims = 0

        if p1 is None and p2 is None:
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

        self.set_center_endpoints(center, p1, p2)

    def set_center_endpoints(self, center, p1=None, p2=None):
        if ((p1 is None and p2 is not None) or
            (p1 is not None and p2 is None)):
            raise GeometryError(
                "Horospherical arc determined by two endpoints and a centerpoint, but"
                " this was not provided"
            )

        if p2 is None:
            proj_data = Point(center).proj_data
        else:
            center_data = Point(center).proj_data
            p1_data = Point(p1).proj_data
            p2_data = Point(p2).proj_data
            proj_data = np.stack([center_data, p1_data, p2_data], axis=-2)

        self.set(proj_data)

    def set(self, proj_data):
        HyperbolicObject.set(self, proj_data)
        self.center = self.proj_data[..., 0, :]
        self.reference = self.proj_data[..., 1, :]
        self.endpoints = self.proj_data[..., 1:, :]

    def circle_parameters(self, model=Model.POINCARE, degrees=True):
        """Get parameters for a circle describing this horospherical arc.

        """
        center, radius = Horosphere.sphere_parameters(self, model=model)
        model_coords = self.endpoint_coords(model)

        thetas = utils.circle_angles(center, model_coords)

        center_theta = utils.circle_angles(
            center, self.center_coords(model=model)
        )[..., 0]

        thetas = np.flip(utils.arc_include(thetas, center_theta), axis=-1)

        if degrees:
            thetas *= 180 / np.pi

        return center, radius, thetas

class BoundaryArc(Geodesic):
    """Model for an arc sitting in the boundary of hyperbolic space.

    """
    def __init__(self, endpoint1, endpoint2=None):
        PointPair.__init__(self, endpoint1, endpoint2)

    def set(self, proj_data):
        HyperbolicObject.set(self, proj_data)
        self.endpoints = proj_data[..., :2, :]
        self.orientation_pt = proj_data[..., 2, :]

    def set_endpoints(self, endpoint1, endpoint2):
        if endpoint2 is None:
            endpoint_data = Point(endpoint1).proj_data
        else:
            pt1 = Point(endpoint1)
            pt2 = Point(endpoint2)
            endpoint_data = np.stack(
                [pt1.proj_data, pt2.proj_data], axis=-2
            )

        self._build_orientation_point(endpoint_data)

    def orientation(self):
        return np.linalg.det(self.proj_data)

    def flip_orientation(self):
        self.proj_data[..., 2, :] *= -1
        self.set(self.proj_data)

    def endpoint_coords(self, model, ordered=True):
        endpoints = Geodesic.endpoint_coords(self, model)
        endpoints[self.orientation() < 0] = np.flip(
            endpoints[self.orientation() < 0],
            axis=-2
        )
        return endpoints

    def _build_orientation_point(self, endpoints):
        n = endpoints.shape[-1]
        orientation_pt_1 = np.zeros(endpoints.shape[:-2] +
                                    (1, n))
        orientation_pt_2 = np.zeros(endpoints.shape[:-2] +
                                    (1, n))
        orientation_pt_1[..., 0, 0] = 1.0
        orientation_pt_2[..., 0, 1] = 1.0

        point_data = np.concatenate((endpoints, orientation_pt_1), axis=-2)

        dets = np.linalg.det(point_data)
        point_data[np.abs(dets) < ERROR_THRESHOLD, 2] = orientation_pt_2

        signs = np.linalg.det(point_data)
        point_data[signs < 0, 2] *= -1

        self.set(point_data)


    def circle_parameters(self, model=Model.POINCARE, degrees=True):
        """Compute parameters (center, radius) for this boundary arc.

        """
        if model != Model.POINCARE and model != Model.KLEIN:
            raise GeometryError(
                ("Cannot compute circular parameters for a boundary arc in model: '{}'"
                 ).format(model)
            )
        center = np.zeros(self.proj_data.shape[:-2] + (2,))
        radius = np.ones(self.proj_data.shape[:-2])

        coords = self.endpoint_coords(model, ordered=True)
        thetas = utils.circle_angles(center, coords)

        if degrees:
            thetas *= 180 / np.pi



        return center, radius, thetas

class Polygon(Point, projective.Polygon):
    """Model for a geodesic polygon in hyperbolic space.

    Underlying data consists of the vertices of the polygon. We also
    keep track of auxiliary data, namely the proj_data of the segments
    making up the edges of the polygon.
    """
    def __init__(self, vertices, aux_data=None):
        self.segment_class = Segment
        HyperbolicObject.__init__(self, vertices, aux_data,
                                  unit_ndims=2, aux_ndims=3)

    def _compute_aux_data(self, proj_data):
        segments = Segment(proj_data, np.roll(proj_data, -1, axis=-2))
        return segments.proj_data

    def get_vertices(self):
        return Point(self.proj_data)

    def get_edges(self):
        return Segment(self.edges)

    def circle_parameters(self, short_arc=True, degrees=True,
                          model=Model.POINCARE, flatten=False):
        if not flatten:
            return self.get_edges().circle_parameters(short_arc, degrees, model)

        flat_segments = self.get_edges().flatten_to_unit()
        return flat_segments.circle_parameters(short_arc, degrees, model)

    def regular_polygon(n, radius=None, angle=None, dimension=2):
        """Get a regular polygon with n vertices, inscribed on a circle of
        radius hyp_radius.

        (This is actually vectorized.)

        """
        if radius is None and angle is None:
            raise ValueError(
                "Must provide either an angle or a radius to regular_polygon"
            )

        if radius is None:
            radius = regular_polygon_radius(n, angle)

        hyp_radius = np.array(radius)
        tangent = TangentVector.get_base_tangent(dimension,
                                                 hyp_radius.shape).normalized()
        start_vertex = tangent.point_along(hyp_radius)

        cyclic_rep = HyperbolicRepresentation()
        cyclic_rep["a"] = Isometry.standard_rotation(2 * np.pi / n, dimension=dimension)

        words = ["a" * i for i in range(n)]
        mats = cyclic_rep.isometries(words)

        vertices = mats.apply(start_vertex, "pairwise_reversed")
        return Polygon(vertices)

    def regular_surface_polygon(g, dimension=2):
        """Get a regular polygon which is the fundamental domain for the
        action of a hyperbolic surface group with genus g.

        """
        return Polygon.regular_polygon(4 * g, radius=genus_g_surface_radius(g),
                                       dimension=dimension)

class Isometry(projective.Transformation, HyperbolicObject):
    """Model for an isometry of hyperbolic space.

    """
    def __init__(self, proj_data, column_vectors=False):
        projective.Transformation.__init__(self, proj_data, column_vectors)

    def _fixpoint_data(self, sort_eigvals=True):

        # find fixpoints in projective space, and their eigenvalues and minkowski norms

        eigvals, eigvecs = np.linalg.eig(self.proj_data.swapaxes(-1, -2))
        norms = utils.normsq(eigvecs.swapaxes(-1, -2),  self.minkowski)

        # 1 for eigenvectors which actually lie in H^n, 0 for outside vectors
        in_plane = np.where(norms > ERROR_THRESHOLD, 0, 1)

        # primary sort key is whether or not we're in the plane,
        # secondary is the eigenvalue modulus
        if sort_eigvals:
            sort_order = np.stack([np.abs(eigvals),
                                   -1 * np.abs(np.imag(eigvals)),
                                   in_plane])
            sort_indices = np.lexsort(sort_order, axis=-1)
            sort_indices = np.expand_dims(sort_indices, axis=-2)
        else:
            sort_indices = np.argsort(in_plane, axis=-1)

        # we want a descending sort to put maximum modulus eigenvalues first
        sort_indices = np.flip(sort_indices, axis=-1)

        pt_data = np.take_along_axis(eigvecs, sort_indices, axis=-1).swapaxes(-1, -2)

        return pt_data

    def _data_to_object(self, data):
        return HyperbolicObject(data)

    def axis(self):
        return Geodesic(self.fixed_point_pair())

    def isometry_type(self):
        fixpoint_data = self._fixpoint_data(sort_eigvals=True)
        fixpoint

    def fixed_point_pair(self, sort_eigvals=True):
        """Find fixed points for this isometry in the closure of hyperbolic space.

        Parameters
        ----------
        sort_eigvals : bool
            If `True`, guarantee that the fixpoints are ordered in
            order of descending eigenvalue moduli

        Returns
        --------
        Point
            Points fixed by this isometry object.

        """
        fixpoint_data = np.real(self._fixpoint_data(sort_eigvals))
        return PointPair(fixpoint_data[..., :2, :])

    def fixed_point(self, max_eigval=True):
        """Find a fixed point for this isometry in the closure of hyperbolic
           space.

        Parameters
        ----------
        max_eigval : bool
            If `True`, guarantee that the eigenvalue for this fixed
            point has maximum modulus

        Returns
        --------
        Point
            A point (possibly ideal) fixed by this isometry object.

        """
        fixpoint_data = np.real(self._fixpoint_data(max_eigval))
        return Point(fixpoint_data[..., 0, :])


    def elliptic(dimension, block_elliptic, column_vectors=True):
        """Get an elliptic isometry stabilizing the origin in the
        Poincare/Klein models.

        block_elliptic is an element of O(n), whose image is taken
        diagonally in O(n,1).

        """
        mat = np.zeros((dimension + 1, dimension + 1))
        mat[0,0] = 1.0
        mat[1:, 1:] = block_elliptic

        return Isometry(mat, column_vectors=column_vectors)

    def standard_loxodromic(dimension, parameter):
        """Get a loxodromic isometry whose axis intersects the origin.

        WARNING: not vectorized.

        """
        basis_change = _loxodromic_basis_change(dimension)
        diagonal_loxodromic = np.diag(
            np.concatenate(([parameter, 1.0/parameter],
                       np.ones(dimension - 1)))
        )

        return Isometry((basis_change @ diagonal_loxodromic @
                        np.linalg.inv(basis_change)),
                        column_vectors=True)

    def standard_rotation(angle, dimension=2):
        """Get a rotation about the origin by a fixed angle.

        WARNING: not vectorized.
        """
        affine = np.identity(dimension)
        affine[0:2, 0:2] = utils.rotation_matrix(angle)

        return Isometry.elliptic(dimension, affine)

class HyperbolicRepresentation(projective.ProjectiveRepresentation):
    """Model a representation for a finitely generated group
    representation into O(n,1).

    Really this is just a convenient way of mapping words in the
    generators to hyperbolic isometries - there's no group theory
    being done here at all.

    """
    def __getitem__(self, word):
        matrix = self._word_value(word)
        return Isometry(matrix, column_vectors=True)

    def normalize(self, matrix):
        dimension = np.array(matrix).shape[-1]
        return utils.indefinite_orthogonalize(minkowski(dimension), matrix)

    def isometries(self, words):
        """Get an Isometry object holding the matrices which are the images of
        a sequence of words in the generators.

        """
        return Isometry(self.transformations(words))

def minkowski(dimension):
    return np.diag(np.concatenate(([-1.0], np.ones(dimension - 1))))

def kleinian_coords(points, column_vectors=False):
    """Get kleinian coordinates for an ndarray of points.

    """
    return projective.affine_coords(points, chart_index=0,
                                    column_vectors=column_vectors)

def get_point(coords, model="klein"):
    return Point(coords, model)

def get_boundary_point(angle):
    return IdealPoint.from_angle(angle)

def hyperboloid_coords(points, column_vectors=False):
    """Project an ndarray of points to the unit hyperboloid defined by the
    Minkowski quadratic form."""

    proj_coords = points
    if column_vectors:
        proj_coords = coords.swapaxes(-1, -2)

    dim = proj_coords.shape[-1]
    hyperbolized = utils.normalize(proj_coords, minkowski(dim))

    if column_vectors:
        hyperbolized = hyperbolized.swapaxes(-1, -2)

    return hyperbolized

def close_to_boundary(vectors):
    """Return true if all of the given vectors have kleinian coords close
    to the boundary of H^n.

    """
    return (np.abs(utils.normsq(kleinian_coords(vectors)) - 1)
            < ERROR_THRESHOLD).all()

def all_spacelike(vectors):
    """Return true if all of the vectors are spacelike.

    Highly susceptible to round-off error, probably don't rely on
    this.

    """
    dim = np.array(vectors).shape[-1]
    return (utils.normsq(vectors, minkowski(dim)) > ERROR_THRESHOLD).all()

def all_timelike(self, vectors):
    """Return true if all of the vectors are timelike.

    Highly susceptible to round-off error, probably don't rely on
    this.

    """
    dim = np.array(vectors).shape[-1]
    return (utils.normsq(vectors, minkowski(dim)) < ERROR_THRESHOLD).all()

def all_lightlike(self, vectors):
    """Return true if all of the vectors are timelike.

    Highly susceptible to round-off error, probably don't rely on
    this.

    """
    dim = np.array(vectors).shape[-1]
    return (np.abs(utils.normsq(vectors, minkowski(dim))) < ERROR_THRESHOLD).all()

def kleinian_to_poincare(points):
    euc_norms = utils.normsq(points)
    #we take absolute value to combat roundoff error
    mult_factor = 1 / (1. + np.sqrt(np.abs(1 - euc_norms)))

    return (points.T * mult_factor.T).T

def poincare_to_kleinian(points):
    euc_norms = utils.normsq(points)
    mult_factor = 2. / (1. + euc_norms)

    return (points.T * mult_factor.T).T

def poincare_to_halfspace(points):
    y = points[..., 0]
    v = points[..., 1:]
    x2 = utils.normsq(v)

    halfspace_coords = np.zeros_like(points)
    denom = (x2 + (y - 1)*(y - 1))

    with np.errstate(divide="ignore", invalid="ignore"):
        halfspace_coords[..., :-1] = (-2. * v) / denom[..., np.newaxis]
        halfspace_coords[..., -1] = (1 - x2 - y * y) / denom

    return halfspace_coords

def halfspace_to_poincare(points):
    y = points[..., -1]
    v = points[..., :-1]
    x2 = utils.normsq(v)

    poincare_coords = np.zeros_like(points)
    denom = (x2 + (y + 1)*(y + 1))
    poincare_coords[..., 1:] = (-2. * v) / denom[..., np.newaxis]
    poincare_coords[..., 0] = (x2 + y * y - 1) / denom

    return poincare_coords

def timelike_to(self, v, force_oriented=False):
    """Find an isometry taking the origin of the Poincare/Klein models to
    the given vector v.

    We expect v to be timelike in order for this to make sense.

    """
    if not all_timelike(v):
        raise GeometryError( "Cannot find isometry taking a"
        " timelike vector to a non-timelike vector."  )

    dim = np.array(vectors).shape[-1]
    return Isometry(utils.find_isometry(minkowski(dim),
                                        v, force_oriented),
                    column_vectors=False)

def sl2r_iso(matrix):
    """Convert 2x2 matrices to isometries.

    matrix is an array of shape (..., 2, 2), and interpreted as an
    array of elements of SL^+-(2, R).

    """

    return Isometry(representation.sl2_to_so21(np.array(matrix)),
                    column_vectors=True)

def project_to_hyperboloid(basepoint, tangent_vector):
    """Project a vector in R^(n,1) to lie in the tangent space to the unit
    hyperboloid at a given basepoint.

    """
    dim = basepoint.shape[-1]
    return tangent_vector - utils.projection(
        tangent_vector, basepoint, minkowski(dim))

def hyp_to_affine_dist(r):
    """Convert distance in hyperbolic space to distance from the origin in
    the Klein model.

    """
    return (np.exp(2 * r) - 1) / (1 + np.exp(2 * r))

def _loxodromic_basis_change(dimension):
    mat = np.identity(dimension + 1)
    basis_change = np.array([
        [1.0, 1.0],
        [1.0, -1.0]
    ])
    mat[0:2, 0:2] = basis_change

    return mat

def regular_polygon_radius(n, interior_angle):
    """Find r such that a regular n-gon inscribed on a circle of radius r
    has the given interior angle.

    """
    alpha = interior_angle / 2
    gamma = np.pi / n
    term = ((np.cos(alpha)**2 - np.sin(gamma)**2) /
            ((np.sin(alpha) * np.sin(gamma))**2))
    return np.arcsinh(np.sqrt(term))

def polygon_interior_angle(n, hyp_radius):
    """Get the interior angle of a regular n-gon inscribed on a circle
    with the given hyperbolic radius.

    """
    gamma = np.pi / n
    denom = np.sqrt(1 + (np.sin(gamma) * np.sinh(hyp_radius))**2)
    return 2 * np.arcsin(np.cos(gamma) / denom)

def genus_g_surface_radius(g):
    """Find the radius of a regular polygon giving the fundamental domain
    for the action of a hyperbolic surface group with genus g.

    """
    return regular_polygon_radius(4 * g, np.pi / (4*g))

def spacelike_to(v, force_oriented=False):
    """Find an isometry taking the second standard basis vector (0, 1, 0,
    ...) to the given vector v.

    We expect v to be spacelike in order for this to make sense.

    """
    dim = np.array(v).shape[-1]

    normed = utils.normalize(v, minkowski(dim))

    if not all_spacelike(v):
        raise GeometryError( "Cannot find isometry taking a"
        " spacelike vector to a non-spacelike vector.")

    iso = utils.find_isometry(minkowski(dim), normed)

    #find the index of the timelike basis vector
    lengths = np.expand_dims(utils.normsq(iso, minkowski(dim)), axis=-1)
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

    return Isometry(p_iso, column_vectors=False)
