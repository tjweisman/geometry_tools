r"""Model objects in hyperbolic space with numerical coordinates.

This module provides abstract versions of various objects in hyperbolic
geometry (points, line segments, polygons, isometries, etc.) and gives a
unified framework for *translating those objects by isometries* and *getting
their coordinates in various models of hyperbolic space.*

For instance, to get a single point in the hyperbolic plane, use the `Point`
class:

```python
from geometry_tools import hyperbolic

# get a point by providing kleinian coordinates
point = hyperbolic.Point([0.1, 0.0], model=hyperbolic.Model.KLEIN)
# or equivalently:
point = hyperbolic.get_point([0.1, 0.0], model="klein")

# get the poincare coordinates of this point.
point.coords(model="poincare")
```
    array([0.05012563, 0.        ])

The `Isometry` class models isometries of hyperbolic space. The easiest way to
obtain an isometry of the hyperbolic plane is to use the isomorphism from
\(\mathrm{PSL}(2, \mathbb{R})\) to the (orientation-preserving) isometry group
 of \(\mathbb{H}^2\).

```python
from geometry_tools import hyperbolic

# get a loxodromic isometry and a point in H^2
hyp_iso = hyperbolic.sl2r_iso([[2., 0.], [0., -1./2]])
point = hyperbolic.get_point([0., 0.])

# apply the isometry to the point
(hyp_iso @ point).coords(model="halfplane")
```
    array([-0.  ,  0.25])


This module is built on top of the `geometry_tools.projective`
submodule, which means that it is possible to easily and efficiently
build *composite* hyperbolic objects out of arrays of subobjects, and
then translate the entire composite object at once. For instance, we
can build an array of points, and translate all of those points by an
isometry:

```python
from geometry_tools import hyperbolic

# make two points in Klein coordinates
p1 = hyperbolic.Point([0., 0.1], model="klein")
p2 = hyperbolic.Point([0.1, 0.], model="klein")

# package these two points together into a single hyperbolic object
# (an array of points)
pts = hyperbolic.Point([p1, p2])

# get a parabolic isometry
iso = hyperbolic.sl2r_iso([[1., 1.], [0., 1.]])

(iso @ pts).coords(model="klein")
```
    array([[-0.375     ,  0.6875    ],
           [-0.29032258,  0.70967742]])

The data is returned as a 2x2 numpy array, giving x,y Kleinian coordinates for
the translated points.

Working with composite objects can be nice because it's a convenient way to
handle *orbits* of hyperbolic objects under groups of isometries. To see
this, we can use the `HyperbolicRepresentation` class to make a representation of
a free group, and then translate a pair of points by some words in the group.

```python
from geometry_tools import hyperbolic
from numpy import pi

# make a free group representation by mapping the generators to loxodromic
# isometries with perpendicular axes
free_rep = hyperbolic.HyperbolicRepresentation()
free_rep["a"] = hyperbolic.sl2r_iso([[3., 0], [0., 1./3]])
rot = hyperbolic.Isometry.standard_rotation(pi / 2)
free_rep["b"] = rot @ free_rep["a"] @ rot.inv()

# make a pair of points in hyperbolic space
pt = hyperbolic.Point([[0., 0.3],
                       [0.1, 0.0]], model="klein")

# get image of reduced words of length at most 3
words = free_rep.free_words_less_than(2)
isos = free_rep.isometries(words)

# get coordinates of points translated by these isometries.
# we have to specify "pairwise" argument because we want each
# isometry to apply to each point in our composite Point object.
(isos.apply(pt, "pairwise")).coords(model="klein")
```
    array([[[ 0.00000000e+00,  3.00000000e-01],
        [ 1.00000000e-01,  0.00000000e+00]],

       [[-9.75609756e-01,  6.58536585e-02],
        [-9.70270270e-01,  0.00000000e+00]],

       [[ 9.75609756e-01,  6.58536585e-02],
        [ 9.80000000e-01,  0.00000000e+00]],

       [[-6.41883840e-17, -9.55172414e-01],
        [ 2.19512195e-02, -9.75609756e-01]],

       [[ 5.73042276e-17,  9.86792453e-01],
        [ 2.19512195e-02,  9.75609756e-01]]])

The coordinate data is returned as a 5x2x2 numpy array, since there are 5
isometries being applied to 2 points, each of which has 2 coordinates. To
flatten the output to a 10x2 array (representing an array of 10 points), we
could use `HyperbolicObject.flatten_to_unit()`.

    """


from copy import copy
from enum import Enum

import numpy as np

from geometry_tools import projective, representation, utils
from geometry_tools.projective import GeometryError

if utils.SAGE_AVAILABLE:
    from geometry_tools.utils import sagewrap

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
        try:
            base_ring = utils.guess_literal_ring(self.proj_data)
            return minkowski(self.dimension + 1, base_ring=base_ring)
        except AttributeError:
            pass

        return minkowski(self.dimension + 1)

    def coords(self, model, proj_data=None, **kwargs):
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
            return self.kleinian_coords(proj_data, **kwargs)
        if model == Model.PROJECTIVE:
            return self.projective_coords(proj_data, **kwargs)

        raise GeometryError(
            "Unimplemented model for an object of type {}: '{}'".format(
                self.__class__.__name__, model
            ))

    def kleinian_coords(self, aff_data=None, **kwargs):
        return self.affine_coords(aff_data, chart_index=0, **kwargs)

class Point(HyperbolicObject, projective.Point):
    """Model for a point (or ndarray of points) in the closure of
    hyperbolic space.

    """
    def __init__(self, point, model=Model.PROJECTIVE, **kwargs):
        """

        Parameters
        ----------
        point : HyperbolicObject or iterable or ndarray
            Data used to construct a point or an array of points. If
            `point` is a `HyperbolicObject`, then use the underlying
            coordinate data to build an array of points. If `point` is
            an iterable of `HyperbolicObject`s, build a composite `Point`
            object out of this array. If `point` is an `ndarray`, then
            interpret this as coordinate data for an array of points
            in some hyperbolic model.
        model : Model
            if `point` is numerical data, then what model of
            hyperbolic space we should use to interpret `point` as
            coordinates.

        """
        self.unit_ndims = 1
        self.aux_ndims = 0
        self.dual_ndims = 0

        try:
            self._construct_from_object(point, **kwargs)
            return
        except TypeError:
            pass

        self.coords(model, point, **kwargs)

    def hyperboloid_coords(self, proj_data=None, **kwargs):
        """Get or set point coordinates in the hyperboloid model.

        Parameters
        ----------
        proj_data : ndarray
            Data to set coordinates to, in the hyperboloid
            model. The last dimension of the array is the
            dimension of the vector space R^(n,1). If `None`, do not
            update coordinates.

        Returns
        -------
        ndarray
            Hyperboloid coordinates for this point (or array of
            points).

        """
        if proj_data is not None:
            self.set(proj_data, **kwargs)

        return hyperboloid_coords(self.proj_data)

    def poincare_coords(self, proj_data=None, **kwargs):
        """Get or set point coordinates in the hyperboloid model.

        Parameters
        ----------
        proj_data : ndarray
            Data to set coordinates to, as points in projective space.
            The last dimension of the array is the dimension of the
            vector space R^(n,1). If `None`, do not update
            coordinates.

        Returns
        -------
        ndarray
            Projective coordinates for this point (or array of
            points).

        """
        if proj_data is not None:
            klein = poincare_to_kleinian(np.array(proj_data))
            self.kleinian_coords(klein, **kwargs)

        return kleinian_to_poincare(self.kleinian_coords())

    def halfspace_coords(self, proj_data=None, **kwargs):
        """Get or set point coordinates in the half-space model.

        Parameters
        ----------
        proj_data : ndarray
            Data to set coordinates to, as points in half-space.  The
            last dimension of the array is the dimension of hyperbolic
            space H^n. If `None`, do not update coordinates.

        Returns
        -------
        ndarray
            Half-space coordinates for this point (or array of
            points).

        """
        poincare = None

        if proj_data is not None:
            poincare = halfspace_to_poincare(np.array(proj_data))

        poincare_coords = self.poincare_coords(poincare, **kwargs)
        return poincare_to_halfspace(poincare_coords)

    def coords(self, model, proj_data=None, **kwargs):
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
            return HyperbolicObject.coords(self, model, proj_data, **kwargs)
        except GeometryError as e:
            if model == Model.POINCARE:
                return self.poincare_coords(proj_data, **kwargs)
            if model == Model.HYPERBOLOID:
                return self.hyperboloid_coords(proj_data, **kwargs)
            if model == Model.HALFSPACE:
                return self.halfspace_coords(proj_data, **kwargs)
            raise e

    def distance(self, other):
        """Compute hyperbolic distances between pairs of points, or pairs of
            arrays of points.

        Parameters
        ----------
        other : Point
            The point to compute distances to

        Returns
        -------
        ndarray
            distances in hyperbolic space between self and other

        """
        #TODO: allow for pairwise distances
        products = utils.apply_bilinear(self.proj_data, other.proj_data,
                                        self.minkowski)

        return np.arccosh(np.abs(products))

    def origin_to(self):
        """Get an isometry taking an "origin" point to this point

        Returns
        -------
        Isometry :
            Some isometry taking the point with Kleinian/Poincare
            coordinates (0, 0, ...) to this point. This isometry is
            not uniquely determined and is not guaranteed to be
            orientation-preserving.

        """
        return timelike_to(self.proj_data)

    def unit_tangent_towards(self, other):
        """Get a unit tangent vector with this basepoint, pointing at another point

        Parameters
        ----------
        other : Point
            another point in hyperbolic space where the unit tangent
            vector points

        Returns
        -------
        TangentVector
            A tangent vector with this point at its base and pointing
            towards `other`.

        """
        diff = other.proj_data - self.proj_data
        return TangentVector(self, diff).normalized()

    def get_origin(dimension, shape=(), **kwargs):
        """Get a point (or ndarray of points) at the "origin" of hyperbolic
           space

        Parameters
        ----------
        dimension : int
            Dimension of hyperbolic space where this point lives
        shape : tuple(int)
            Shape of array of origin points to get

        Returns
        -------
        Point
            Point object with Kleinian/Poincare coords (0, 0, ...)
        """

        return Point(utils.zeros(shape + (dimension,), **kwargs),
                     model="klein")

class DualPoint(Point):
    """Model for a "dual point" in hyperbolic space (a point in the
    complement of the projectivization of the Minkowski light cone,
    corresponding to a geodesic hyperplane in hyperbolic space).

    """

    def _assert_geometry_valid(self, proj_data):
        Point._assert_geometry_valid(self, proj_data)
        if not CHECK_LIGHT_CONE:
            return

        if not spacelike(proj_data).all():
            raise GeometryError("Dual point data must consist of vectors"
                                " in the complement of the Minkowski light cone")

class IdealPoint(Point):
    """Model for an ideal point in hyperbolic space (lying on the boundary
    of the projectivization of the Minkowski light cone)

    """
    def _assert_geometry_valid(self, proj_data):
        Point._assert_geometry_valid(self, proj_data)

        if not CHECK_LIGHT_CONE:
            return

        if not lightlike(proj_data).all():
            raise GeometryError("Ideal point data must consist of vectors"
                                "in the boundary of the Minkowski light cone")

    @staticmethod
    def from_angle(theta, dimension=2, base_ring=None, dtype=None):
        if dimension < 2:
            raise GeometryError(
                "Cannot use an angle to specify a point on the boundary of"
                " hyperbolic 1-space"
            )

        like = theta
        if dtype is not None:
            like = None

        one = utils.number(1, like=like, dtype=dtype, base_ring=base_ring)

        result = utils.zeros(np.array(theta).shape + (dimension + 1,),
                             like=like, dtype=dtype, base_ring=base_ring)

        result[..., 0] = one
        result[..., 1] = np.cos(theta)
        result[..., 2] = np.sin(theta)

        pt = IdealPoint(result)

        if base_ring is not None:
            pt = pt.change_base_ring(base_ring)

        return pt

class Subspace(IdealPoint):
    """Model for a geodesic subspace of hyperbolic space.

    """
    def __init__(self, proj_data, **kwargs):
        """
        Parameters
        ----------
        proj_data : HyperbolicObject or ndarray
            Data used to construct a subspace (or array of
            subspaces). If `proj_data` is a `HyperbolicObject`, then
            use underlying coordinate data to build an array of
            subspaces. If `proj_data` is an `ndarray`, then interpret
            this as coordinate data for an array of subspaces in the
            projective model.

        """
        HyperbolicObject.__init__(self, proj_data, unit_ndims=2, **kwargs)

    @property
    def ideal_basis(self):
        return self.proj_data

    def ideal_basis_coords(self, model=Model.KLEIN):
        """Get coordinates for a basis of this subspace lying in the ideal
           boundary of hyperbolic space

        Parameters
        ----------
        model : Model
            Model of hyperbolic space in which to compute coordinates

        Returns
        -------
        ndarray
            Coordinates for k vectors in the ideal boundary of
            hyperbolic space, giving a basis for this subspace.

        """
        #we don't return proj_data directly, since we want subclasses
        #to be able to use this method even if the structure of the
        #underlying data is different.
        return Point(self.ideal_basis).coords(model)

    def spacelike_complement(self, compute_exact=True):
        """Get a spacelike point in the orthogonal complement to this subspace

        Returns
        -------
        DualPoint
            A spacelike (i.e. Minkowski-positive) point which is
            Minkowski-orthogonal to this subspace. This is not
            uniquely determined (as a point in projective space)
            unless the subspace is a hyperplane.

        """

        orthed = self._data_with_dual(compute_exact=compute_exact)
        return DualPoint(orthed[..., 0, :])

    def _data_with_dual(self, compute_exact=True):
        midpoints = np.sum(self.ideal_basis, axis=-2) / self.ideal_basis.shape[-2]

        poincare_ctr, poincare_rad = self.sphere_parameters(model=Model.POINCARE)
        spacelike_guess = Point(poincare_ctr, model=Model.KLEIN).coords(
            model=Model.PROJECTIVE
        )

        to_orthogonalize = np.concatenate(
            [np.expand_dims(midpoints, axis=-2),
            self.ideal_basis[..., 1:, :],
            np.expand_dims(spacelike_guess, axis=-2)],
            axis=-2)

        orthed = utils.indefinite_orthogonalize(self.minkowski,
                                                to_orthogonalize,
                                                compute_exact=compute_exact)
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
        -------
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

    def reflection_across(self, compute_exact=True):
        """Get a hyperbolic isometry reflecting across this hyperplane.

        Returns
        -------
        Isometry
            Isometry reflecting across this hyperplane.

        """
        dual_data = self._data_with_dual(compute_exact=compute_exact)
        if self.dimension + 1 != dual_data.shape[-2]:
            raise GeometryError(
                ("Cannot compute a reflection across a subspace of "
                 "dimension {}").format(dual_data.shape[-2])
            )

        refdata = (utils.invert(dual_data) @
                   self.minkowski @
                   dual_data)

        return Isometry(refdata, column_vectors=False)

class PointPair(Point, projective.PointPair):
    """Abstract model for a hyperbolic object whose underlying data is
    determined by a pair of points in R^(n,1)

    """
    def __init__(self, endpoint1, endpoint2=None, **kwargs):
        """If `endpoint2` is `None`, interpret `endpoint1` as either a (2
        x...x n) `ndarray` (where n is the dimension of the underlying
        vector space), or else a composite `Point` object which can be
        unpacked into two Points (which may themselves be composite).

        If `endpoint2` is given, then both `endpoint1` and `endpoint2`
        can be used to construct `Point` objects, which serve as the
        endpoints for this pair of points.

        Parameters
        ----------
        endpoint1 : Point or ndarray
            One (or both) endpoints of the point pair
        endpoint2 : Point or ndarray
            The other endpoint of the point pair. If `None`,
            `endpoint1` contains the data for both points in the pair.

        """

        projective.PointPair.__init__(self, endpoint1, endpoint2, **kwargs)

    def endpoint_coords(self, model=Model.KLEIN):
        """Get coordinates for the endpoints of this point pair

        Parameters
        ----------
        model : Model
            model to compute coordinates in

        Returns
        -------
        ndarray
            Coordinates for the endpoints of this PointPair, as an
            ndarray with shape `(2, ... n)` (where `n` is the
            dimension of the hyperbolic space).

        """
        return self.get_endpoints().coords(model)

    def get_endpoints(self):
        """Convert this pair of points to a composite `Point` object

        Returns
        -------
        Point
            Composite `Point` object with the same underlying data as
            this point pair.

        """
        return Point(self.endpoints)

    def get_end_pair(self, as_points=False):
        """Get a pair of `Point` objects, one for each endpoint of this PointPair

        Parameters
        ----------
        as_points : bool
            If `True`, return a pair of `Point` objects. Otherwise,
            return a pair of `ndarray`s with projective coordinates
            for these points.

        Returns
        -------
        tuple
            A tuple `(p1, p2)`, where `p1` and `p2` are either both
            `Point`s or both `ndarrays`, representing the endpoints of
            this pair.

    """
        if as_points:
            p1, p2 = self.get_end_pair(as_points=False)
            return (Point(p1), Point(p2))

        return (self.endpoints[..., 0, :], self.endpoints[..., 1, :])

class Geodesic(PointPair, Subspace):
    """Model for a bi-infinite gedoesic in hyperbolic space.

    """
    @property
    def endpoints(self):
        return self.proj_data[..., :2, :]

    @property
    def ideal_basis(self):
        return self.proj_data[..., :2, :]

    def circle_parameters(self, degrees=True, model=Model.POINCARE):
        """Get parameters describing a circular arc corresponding to this
        geodesic in the Poincare or halfspace models.

        Parameters
        ----------
        degrees : bool
            if `True`, return angles in degrees. Otherwise, return
            angles in radians
        model : Model
            hyperbolic model to use for the computation

        Returns
        -------
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

        pi = utils.pi(like=thetas)
        if degrees:
            thetas *= 180 / pi

        return center, radius, thetas

    @staticmethod
    def from_reflection(reflection):
        """Construct a geodesic which is the fixpoint set of a reflection.

        Parameters
        ----------
        reflection : Isometry or ndarray
            The reflection to compute the fixed geodesic for

        Raises
        ------
        GeometryError :
            If the given isometry is not a reflection, or if the
            dimension of the underlying hyperbolic space is not 2.

        Returns
        -------
        Geodesic
            A `Geodesic` fixed by the given isometry.

    """
        if reflection.dimension != 2:
            raise GeometryError("Creating segment from reflection expects dimension 2, got dimension {}".format(reflection.dimension))

        hyperplane = Hyperplane.from_reflection(reflection)
        pt1 = hyperplane.ideal_basis[..., 0, :]
        pt2 = hyperplane.ideal_basis[..., 1, :]
        return Geodesic(pt1, pt2)

class Segment(Geodesic):
    """Model a geodesic segment in hyperbolic space."""

    def __init__(self, endpoint1, endpoint2=None, aux_ndims=2,
                 **kwargs):

        PointPair.__init__(self, endpoint1, endpoint2,
                           aux_ndims=aux_ndims, **kwargs)

        """self.unit_ndims = 2
        self.aux_ndims = 0
        self.dual_ndims = 0

        if endpoint2 is None:
            try:
                self._construct_from_object(endpoint1, **kwargs)
                return
            except (AttributeError, TypeError, GeometryError):
                pass

            try:
                self.set(endpoint1, **kwargs)
                return
            except (AttributeError, GeometryError) as e:
                pass

        self.set_endpoints(endpoint1, endpoint2, **kwargs)

    def set_endpoints(self, endpoint1, endpoint2=None, **kwargs):
        # reimplemented to also compute ideal endpoints
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
        self._set_optional(**kwargs)"""

    @property
    def ideal_basis(self):
        return self.aux_data

    def _assert_aux_valid(self, aux_data):
        if (aux_data.shape[-2] != 2):
            raise GeometryError( ("Underlying auxiliary data for a hyperbolic"
            " segment must have shape (..., 2, n) but data"
            " has shape {}").format(proj_data.shape))

    def _assert_geometry_valid(self, proj_data):
        HyperbolicObject._assert_geometry_valid(self, proj_data)

        if (proj_data.shape[-2] != 2):
            raise GeometryError( ("Underlying data for a hyperbolic"
            " segment must have shape (..., 2, n) but data"
            " has shape {}").format(proj_data.shape))

        if not CHECK_LIGHT_CONE:
            return

        if not timelike(proj_data[..., 0, :, :]).all():
            raise GeometryError( "segment data at index [..., 0, :,"
            ":] must consist of timelike vectors" )

        if not lightlike(proj_data[..., 1, :, :]).all():
            raise GeometryError( "segment data at index [..., 1, :,"
            ":] must consist of lightlike vectors" )

    def _compute_aux_data(self, end_data):
        base_ring = utils.guess_literal_ring(end_data)
        dim = end_data.shape[-1]

        products = end_data @ minkowski(
            dim, base_ring=base_ring
        ) @ end_data.swapaxes(-1, -2)
        a11 = products[..., 0, 0]
        a22 = products[..., 1, 1]
        a12 = products[..., 0, 1]

        a = a11 - 2 * a12 + a22
        b = 2 * a12 - 2 * a22
        c = a22

        mu1 = (-b + np.sqrt(b * b - 4 * a * c)) / (2*a)
        mu2 = (-b - np.sqrt(b * b - 4 * a * c)) / (2*a)

        null1 = (mu1[..., np.newaxis] * end_data[..., 0, :] +
                 (1 - mu1)[..., np.newaxis] * end_data[..., 1, :])

        null2 = (mu2[..., np.newaxis] * end_data[..., 0, :] +
                 (1 - mu2)[..., np.newaxis] * end_data[..., 1, :])

        ideal_basis = np.stack([null1, null2], axis=-2)

        return ideal_basis

    def geodesic(self):
        """Get the bi-infinite geodesic spanned by this segment

        Returns
        -------
        Geodesic
            Geodesic in hyperbolic space spanned by this segment.
        """
        return Geodesic(self.ideal_basis)

    def ideal_endpoint_coords(self, model=Model.KLEIN):
        """Alias for Subspace.ideal_basis_coords.

        """
        return self.ideal_basis_coords(model)

    def circle_parameters(self, degrees=True, model=Model.POINCARE):
        """Get parameters describing a circular arc corresponding to this
        segment in the Poincare or halfspace models.

        Parameters
        ----------
        degrees : bool
            if `True`, return angles in degrees. Otherwise, return
            angles in radians
        model : Model
            hyperbolic model to use for the computation

        Returns
        -------
        tuple
            tuple `(centers, radii, thetas)`, where `centers`,
            `radii`, and `thetas` are `ndarray`s representing the
            centers, radii, and begin/end angles for the circle
            corresponding to this arc in the given model of hyperbolic
            space. Angles are always specified in counterclockwise
            order.

        """
        center, radius = self.sphere_parameters(model)

        endpoints = self.endpoint_coords(model)

        thetas = utils.circle_angles(center, endpoints)

        if model == Model.POINCARE:
            thetas = utils.short_arc(thetas)
        elif model == Model.HALFSPACE:
            thetas = utils.right_to_left(thetas)

        pi = utils.pi(like=thetas)
        if degrees:
            thetas *= 180 / pi

        return center, radius, thetas


class Hyperplane(Subspace):
    """Model for a geodesic hyperplane in hyperbolic space."""

    #TODO: reimplement so ideal_basis is aux_data
    def __init__(self, hyperplane_data, **kwargs):
        self.unit_ndims = 2
        self.aux_ndims = 0
        self.dual_ndims = 0

        try:
            self._construct_from_object(hyperplane_data, **kwargs)
            return
        except (TypeError, GeometryError):
            pass

        try:
            self.set(hyperplane_data, **kwargs)
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

        if not spacelike(proj_data[..., 0, :]).all():
            raise GeometryError( ("Hyperplane data at index [..., 0, :]"
                                  " must consist of spacelike vectors") )

        if not lightlike(proj_data[..., 1:, :]).all():
            raise GeometryError( ("Hyperplane data at index [..., 1, :]"
                                  " must consist of lightlike vectors") )

    def _data_with_dual(self, **kwargs):
        return self.proj_data

    @property
    def spacelike_vector(self):
        return self.proj_data[..., 0, :]

    @property
    def ideal_basis(self):
        return self.proj_data[..., 1:, :]

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

    @staticmethod
    def from_reflection(reflection):
        """Construct a hyperplane which is the fixpoint set of a reflection.

        Parameters
        ----------
        reflection : Isometry or ndarray
            The reflection to compute the fixed plane for

        Raises
        ------
        GeometryError :
            If the given isometry is not a reflection.

        Returns
        -------
        Hyperplane
            A `Hyperplane` fixed by the given isometry.
        """
        try:
            matrix = reflection.matrix.swapaxes(-1, -2)
        except AttributeError:
            matrix = reflection

        # TODO: make this compatible with sage

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

        # SAGETEST
        spacelike = np.take_along_axis(
            np.real(evecs), np.expand_dims(reflected, axis=(-1,-2)), axis=-1
        )

        return Hyperplane(spacelike.swapaxes(-1,-2))

class TangentVector(PointPair):
    """Model for a tangent vector in hyperbolic space."""
    def __init__(self, point_data, vector=None,
                 unit_ndims=2, aux_ndims=2, dual_ndims=0, **kwargs):
        """If `vector` is `None`, interpret `point_data` as either an ndarray
        of shape `(..., 2, n)` (where `n` is the dimension of the
        underlying vector space), or else a composite HyperbolicObject
        whose data can be unpacked into a point in projective space
        and a tangent vector to a hyperboloid.

        If `vector` is given, then `point_data` is used to construct
        the basepoint for this tangent vector, and `vector` is used to
        construct the tangent vector to the hyperboloid.

        Parameters
        ----------
        point_data : Point or ndarray
            The basepoint of the tangent vector, or (if `vector` is
            `None`) the data of the basepoint and the tangent vector.
        vector : HyperbolicObject or ndarray
            The tangent vector data. If `None`, `point_data` contains
            the data for the basepoint and the tangent vector.

        """
        PointPair.__init__(self, point_data, vector,
                           unit_ndims=unit_ndims,
                           aux_ndims=aux_ndims,
                           dual_ndims=dual_ndims,
                           **kwargs)

    def _assert_geometry_valid(self, proj_data):
        HyperbolicObject._assert_geometry_valid(self, proj_data)

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

        if not timelike(point_data).all():
            raise GeometryError("tangent vector data at index [..., 0, :] must"
                                " consist of timelike vectors")

        products = utils.apply_bilinear(point_data, vector_data, self.minkowski)
        if (np.abs(products) > ERROR_THRESHOLD).any():
            raise GeometryError("tangent vector must be orthogonal to point")

    def _compute_aux_data(self, proj_data):
        point_data = proj_data[..., 0, :]
        vec_data = proj_data[..., 1, :]

        # _compute_aux_data doesn't assume that proj_data has already
        # been set, so we don't refer to self.minkowski here
        base_ring = utils.guess_literal_ring(proj_data)

        projected = project_to_hyperboloid(
            point_data, vec_data,
            minkowski(point_data.shape[-1], base_ring)
        )

        return np.stack([point_data, projected], axis=-2)

    @property
    def point(self):
        return self.proj_data[..., 0, :]

    @property
    def vector(self):
        return self.aux_data[..., 1, :]

    def normalized(self):
        """Get a unit tangent vector in the same direction as this tangent
        vector.

        Returns
        -------
        TangentVector
            Normalized version of this tangent vector (with respect to
            the standard Minkowski inner product)

        """
        normed_vec = utils.normalize(self.vector, self.minkowski)

        return TangentVector(self.point, normed_vec)

    def origin_to(self, force_oriented=True, compute_exact=True):
        """Get an isometry taking the "origin" to this vector.

        The "origin" is a tangent vector whose basepoint lies at (0,
        0, ....) in Poincare or Kleinian coordinates, and whose
        tangent vector is parallel to the second standard basis
        vector.

        Parameters
        ----------
        force_oriented : bool
            If `True`, force the computed isometry to be
            orientation-preserving

        Returns
        -------
        Isometry
            Isometry taking the "origin" to this tangent vector.

        """
        normed = utils.normalize(self.aux_data, self.minkowski)
        isom = utils.find_isometry(self.minkowski, normed,
                                   force_oriented,
                                   compute_exact=compute_exact)

        return Isometry(isom, column_vectors=False)

    def isometry_to(self, other, **kwargs):
        """Get an isometry taking this tangent vector to a scalar multiple of
        the other.

        Parameters
        ----------
        other : TangentVector
            Tangent vector to translate this vector to
        force_oriented : bool
            If `True`, force the computed isometry to be
            orientation-preserving.
        compute_exact : bool
            If True, sage is available, and the underlying dtype of
            both objects supports exact computation, then do
            computations using sage instead of numpy.

        Returns
        -------
        Isometry
            Isometry this tangent vector to `other`.

    """
        return other.origin_to(**kwargs) @ self.origin_to(**kwargs).inv()

    def angle(self, other):
        """Compute the angle between two tangent vectors.

        Parameters
        ----------
        other : TangentVector

        Returns
        -------
        float or ndarray
            The angle between `self` and `other`.

        """
        v1 = project_to_hyperboloid(self.point, self.normalized().vector)
        v2 = project_to_hyperboloid(self.point, other.normalized().vector)

        product = utils.apply_bilinear(v1, v2, self.minkowski)
        return np.arccos(product)

    def point_along(self, distance):
        """Get a point in hyperbolic space along the geodesic specified by
        this tangent vector.

        Parameters
        ----------
        distance : float or ndarray

        Returns
        -------
        Point
            Point in hyperbolic space along the geodesic ray defined
            by this tangent vector

        """
        kleinian_shape = list(self.point.shape)
        kleinian_shape[-1] -= 1

        kleinian_pt = utils.zeros(kleinian_shape,
                                  like=self.proj_data)

        kleinian_pt[..., 0] = hyp_to_affine_dist(distance)

        basepoint = Point(kleinian_pt, model=Model.KLEIN)

        return self.origin_to().apply(basepoint, "elementwise")

    @staticmethod
    def get_base_tangent(dimension, shape=(), **kwargs):
        """Get an "origin" tangent vector.

        The basepoint of this tangent vector has coordinates (0, 0,
        ...) in Poincare/Klein coordinates, and the vector is parallel
        to the second standard basis vector in R^(n,1).

        Parameters
        ----------
        dimension : int
            Dimension of the hyperbolic space where this tangent vector lives
        shape : tuple
            Shape of the composite TangentVector object to return.

        Returns
        -------
        TangentVector
            An "origin" tangent vector.

        """

        origin = Point.get_origin(dimension, shape, **kwargs)
        vector = utils.zeros((dimension + 1,), **kwargs)

        one = utils.number(1, **kwargs)
        vector[..., 1] = one

        return TangentVector(origin, vector, **kwargs)

class Horosphere(HyperbolicObject):
    """Model for a horosphere in hyperbolic space.

    """
    def __init__(self, center, reference_point=None, **kwargs):
        """If `reference_point` is `None`, interpret `center` as either:

        - a (..., 2, n) `ndarray`, where first row of the last two
        ndindices gives the (ideal) center point for this horosphere,
        and the second row gives some reference point on the
        horosphere itself, or

        - a `HyperbolicObject` whose underlying data has the form above.

        If `reference_point` is given, then `center` can be used to
        construct an `IdealPoint` giving the center of the horosphere,
        and `reference_point` can be used to construct a `Point`
        giving some point on the horosphere.

        Parameters
        ----------
        center : Point or ndarray
            The (ideal) center point for this horosphere, or an object
            which can be unpacked into all of the data for the
            horosphere
        reference_point : Point or ndarray
            Any point lying on the horosphere itself. If `None`, then
            `center` contains the data for the reference point as
            well.

        """
        self.unit_ndims = 2
        self.aux_ndims = 0
        self.dual_ndims = 0

        if reference_point is None:
            try:
                self._construct_from_object(center, **kwargs)
                return
            except TypeError:
                pass

        self.set_center_ref(center, reference_point, **kwargs)

    @property
    def center(self):
        return self.proj_data[..., 0, :]

    @property
    def reference(self):
        return self.proj_data[..., 1, :]

    def set_center_ref(self, center, reference_point=None, **kwargs):
        if reference_point is None:
            proj_data = Point(center).proj_data
        else:
            center = IdealPoint(center)
            ref = Point(reference_point)
            proj_data = np.stack([center.proj_data, ref.proj_data], axis=-2)

        self.set(proj_data, **kwargs)

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
        midpt = Point((geodesic_endpts[..., 0, :] + geodesic_endpts[..., 1, :]) / 2,
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
    def __init__(self, center, p1=None, p2=None, **kwargs):
        self.unit_ndims = 2
        self.aux_ndims = 0
        self.dual_ndims = 0

        if p1 is None and p2 is None:
            try:
                self._construct_from_object(center, **kwargs)
                return
            except TypeError:
                pass
            try:
                self.set(center, **kwargs)
                return
            except (AttributeError, GeometryError):
                pass

        self.set_center_endpoints(center, p1, p2, **kwargs)

    def set_center_endpoints(self, center, p1=None, p2=None, **kwargs):
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

        self.set(proj_data, **kwargs)

    @property
    def center(self):
        return self.proj_data[..., 0, :]

    @property
    def reference(self):
        return self.proj_data[..., 1, :]

    @property
    def endpoints(self):
        return self.proj_data[..., 1:, :]

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

        pi = utils.pi(like=thetas)
        if degrees:
            thetas *= 180 / pi

        return center, radius, thetas

class BoundaryArc(Geodesic):
    """Model for an arc sitting in the boundary of hyperbolic space.

    """
    def __init__(self, endpoint1, endpoint2=None, **kwargs):
        PointPair.__init__(self, endpoint1, endpoint2, **kwargs)

    @property
    def endpoints(self):
        return self.proj_data[..., :2, :]

    @property
    def orientation_pt(self):
        return self.proj_data[..., 2, :]

    def set_endpoints(self, endpoint1, endpoint2, **kwargs):
        if endpoint2 is None:
            endpoint_data = Point(endpoint1).proj_data
        else:
            pt1 = Point(endpoint1)
            pt2 = Point(endpoint2)
            endpoint_data = np.stack(
                [pt1.proj_data, pt2.proj_data], axis=-2
            )

        self._build_orientation_point(endpoint_data)
        self._set_optional(**kwargs)

    def orientation(self):
        return utils.det(self.proj_data)

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

        # force numerical approximation for determinants, since we're
        # only using them to find point determining the orientation of
        # the arc
        dets = utils.det(point_data).astype('float64')

        point_data[np.abs(dets) < ERROR_THRESHOLD, 2] = orientation_pt_2

        signs = utils.det(point_data).astype('float64')
        point_data[dets < 0, 2] *= -1

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

        # WRAPLITERAL
        if degrees:
            thetas *= 180 / np.pi



        return center, radius, thetas

class Polygon(Point, projective.Polygon):
    """Model for a geodesic polygon in hyperbolic space.

    Underlying data consists of the vertices of the polygon. We also
    keep track of auxiliary data, namely the proj_data of the segments
    making up the edges of the polygon.
    """
    def __init__(self, vertices, aux_data=None, **kwargs):
        self.segment_class = Segment
        HyperbolicObject.__init__(self, vertices, aux_data,
                                  unit_ndims=2, aux_ndims=3,
                                  **kwargs)

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

    @staticmethod
    def regular_polygon(n, radius=None, angle=None, dimension=2,
                        base_ring=None, **kwargs):
        """Get a regular polygon with n vertices, inscribed on a circle of
        radius hyp_radius.

        (This is actually vectorized.)

        """
        if radius is None and angle is None:
            raise ValueError(
                "Must provide either an angle or a radius to regular_polygon"
            )

        compute_exact = (utils.SAGE_AVAILABLE and base_ring is not None)

        pi = np.pi
        if compute_exact:
            pi = sagewrap.pi

        if radius is None:
            radius = regular_polygon_radius(n, angle)

        hyp_radius = np.array(radius)
        tangent = TangentVector.get_base_tangent(dimension,
                                                 hyp_radius.shape,
                                                 base_ring=base_ring).normalized()

        start_vertex = tangent.point_along(hyp_radius)

        cyclic_rep = HyperbolicRepresentation()
        cyclic_rep["a"] = Isometry.standard_rotation(2 * pi / n, dimension=dimension)

        words = ["a" * i for i in range(n)]
        mats = cyclic_rep.isometries(words)

        vertices = mats.apply(start_vertex, "pairwise_reversed")

        return Polygon(vertices, **kwargs)

    @staticmethod
    def regular_surface_polygon(g, base_ring=None, **kwargs):
        """Get a regular polygon which is the fundamental domain for the
        action of a hyperbolic surface group with genus g.

        """

        compute_exact = (utils.SAGE_AVAILABLE and base_ring is not None)
        radius = genus_g_surface_radius(g, compute_exact=compute_exact)

        if base_ring is not None:
            radius = base_ring(radius)
        return Polygon.regular_polygon(
            4 * g, radius=radius, base_ring=base_ring, **kwargs
        )

class Isometry(projective.Transformation, HyperbolicObject):
    """Model for an isometry of hyperbolic space.

    """
    def __init__(self, proj_data, column_vectors=False, **kwargs):
        projective.Transformation.__init__(self, proj_data, column_vectors,
                                           **kwargs)

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
        ...

    def fixed_point_pair(self, sort_eigvals=True):
        """Find fixed points for this isometry in the closure of hyperbolic space.

        Parameters
        ----------
        sort_eigvals : bool
            If `True`, guarantee that the fixpoints are ordered in
            order of descending eigenvalue moduli

        Returns
        -------
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
        -------
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

        # WRAPLITERAL
        mat = utils.zeros((dimension + 1, dimension + 1),
                          like=block_elliptic)

        # add one to preserve base_ring
        mat[0,0] += 1
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
                        utils.invert(basis_change)),
                        column_vectors=True)

    # TODO: specify base ring for rotation
    def standard_rotation(angle, dimension=2):
        """Get a rotation about the origin by a fixed angle.

        WARNING: not vectorized.
        """

        affine = utils.identity(dimension, like=angle)
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

    def automaton_accepted(self, automaton, length,
                           with_words=False, **kwargs):

        result = projective.ProjectiveRepresentation.automaton_accepted(
            self, automaton, length, with_words=with_words, **kwargs)

        if with_words:
            transformations, words = result
            return (Isometry(transformations), words)

        return Isometry(result)

def minkowski(dimension, base_ring=None):
    form = utils.identity(dimension, base_ring=base_ring)
    form[0, 0] = form[0,0] * -1
    return form

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

def spacelike(vectors):
    """Determine if a vector in R^(n,1) is spacelike.

    Parameters
    ----------
    vectors : ndarray
        Vectors to test for spacelike norms. The last index of the
        array is the dimension of the vector space.

    Returns
    -------
    ndarray(bool)
        `True` for vectors which have Minkowski norm above a fixed
        error threshold.

    """
    dim = np.array(vectors).shape[-1]
    return (utils.normsq(vectors, minkowski(dim)) > ERROR_THRESHOLD).all()

def timelike(vectors):
    """Determine if a vector in R^(n,1) is timelike.

    Parameters
    ----------
    vectors : ndarray
        Vectors to test for timelike norms. The last index of the
        array is the dimension of the vector space.

    Returns
    -------
    ndarray(bool)
        `True` for vectors which have Minkowski norm below a fixed
        error threshold.

    """
    dim = np.array(vectors).shape[-1]
    return (utils.normsq(vectors, minkowski(dim)) < ERROR_THRESHOLD)

def lightlike(vectors):
    """Determine if a vector in R^(n,1) is lightlike.

    Parameters
    ----------
    vectors : ndarray
        Vectors to test for lightlike norms. The last index of the
        array is the dimension of the vector space.

    Returns
    -------
    ndarray(bool)
        `True` for vectors where the absolute value of the Minkowski
        norm is below a fixed error threshold.

    """

    dim = np.array(vectors).shape[-1]
    return (np.abs(utils.normsq(vectors, minkowski(dim))) < ERROR_THRESHOLD)


def kleinian_to_poincare(points):
    euc_norms = utils.normsq(points)
    #we take absolute value to combat roundoff error
    mult_factor = 1 / (1 + np.sqrt(np.abs(1 - euc_norms)))

    return (points.T * mult_factor.T).T

def poincare_to_kleinian(points):
    euc_norms = utils.normsq(points)
    mult_factor = 2 / (1 + euc_norms)

    return (points.T * mult_factor.T).T

def poincare_to_halfspace(points):
    y = points[..., 0]
    v = points[..., 1:]
    x2 = utils.normsq(v)

    halfspace_coords = np.zeros_like(points)
    denom = (x2 + (y - 1)*(y - 1))

    with np.errstate(divide="ignore", invalid="ignore"):
        halfspace_coords[..., :-1] = (-2 * v) / denom[..., np.newaxis]
        halfspace_coords[..., -1] = (1 - x2 - y * y) / denom

    return halfspace_coords

def halfspace_to_poincare(points):
    y = points[..., -1]
    v = points[..., :-1]
    x2 = utils.normsq(v)

    poincare_coords = np.zeros_like(points)
    denom = (x2 + (y + 1)*(y + 1))
    poincare_coords[..., 1:] = (-2 * v) / denom[..., np.newaxis]
    poincare_coords[..., 0] = (x2 + y * y - 1) / denom

    return poincare_coords

def timelike_to(v, force_oriented=False, compute_exact=True):
    """Find an isometry taking the origin of the Poincare/Klein models to
    the given vector v.

    We expect v to be timelike in order for this to make sense.

    """
    if not timelike(v).all():
        raise GeometryError( "Cannot find isometry taking a"
        " timelike vector to a non-timelike vector."  )

    dim = np.array(v).shape[-1]
    return Isometry(utils.find_isometry(minkowski(dim),
                                        v, force_oriented,
                                        compute_exact=compute_exact),
                    column_vectors=False)

def sl2r_iso(matrix):
    """Convert 2x2 matrices to isometries.

    matrix is an array of shape (..., 2, 2), and interpreted as an
    array of elements of SL^+-(2, R).

    """

    return Isometry(representation.sl2_to_so21(np.array(matrix)),
                    column_vectors=True)

def project_to_hyperboloid(basepoint, tangent_vector, form=None):
    """Project a vector in R^(n,1) to lie in the tangent space to the unit
    hyperboloid at a given basepoint.

    """
    if form is None:
        form = minkowski(basepoint.shape[-1])

    return tangent_vector - utils.projection(
        tangent_vector, basepoint, form
    )

def hyp_to_affine_dist(r):
    """Convert distance in hyperbolic space to distance from the origin in
    the Klein model.

    """
    return (np.exp(2 * r) - 1) / (1 + np.exp(2 * r))

def _loxodromic_basis_change(dimension):
    # WRAPLITERAL
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
    pi = utils.pi(like=interior_angle)

    alpha = interior_angle / 2
    gamma = pi / n
    term = ((np.cos(alpha)**2 - np.sin(gamma)**2) /
            ((np.sin(alpha) * np.sin(gamma))**2))

    return np.arcsinh(np.sqrt(term))

def polygon_interior_angle(n, hyp_radius):
    """Get the interior angle of a regular n-gon inscribed on a circle
    with the given hyperbolic radius.

    """
    pi = utils.pi(like=hyp_radius)

    gamma = pi / n
    denom = np.sqrt(1 + (np.sin(gamma) * np.sinh(hyp_radius))**2)
    return 2 * np.arcsin(np.cos(gamma) / denom)

def genus_g_surface_radius(g, compute_exact=True):
    """Find the radius of a regular polygon giving the fundamental domain
    for the action of a hyperbolic surface group with genus g.

    """

    pi = utils.pi(compute_exact)

    return regular_polygon_radius(4 * g, pi / (2*g))

def spacelike_to(v, force_oriented=False, compute_exact=True):
    """Find an isometry taking the second standard basis vector (0, 1, 0,
    ...) to the given vector v.

    We expect v to be spacelike in order for this to make sense.

    """
    dim = np.array(v).shape[-1]

    normed = utils.normalize(v, minkowski(dim))

    if not spacelike(v).all():
        raise GeometryError( "Cannot find isometry taking a"
        " spacelike vector to a non-spacelike vector.")

    iso = utils.find_isometry(minkowski(dim), normed,
                              compute_exact=compute_exact)

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

def identity(dimension):
    return Isometry(np.identity(dimension + 1))
