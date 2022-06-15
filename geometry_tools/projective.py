"""Work with projective space in numerical coordinates.

The main class provided by this module is ProjectiveSpace, which
represents a copy of n-dimensional projective space.

"""

from copy import copy

import numpy as np
from scipy.spatial import ConvexHull

from geometry_tools import utils
from geometry_tools import representation


class GeometryError(Exception):
    """Thrown if there's an attempt to construct a geometric object with
    numerical data that doesn't make sense for that type of object.

    """
    pass

class ProjectiveObject:
    """Represent some object in projective geometry (possibly a composite
    object).

    The underlying data of a projective object is stored as a numpy
    ndarray. The last `unit_ndims` ndims of this array describe a
    *single* instance of this type of object.

    For example, a `Polygon` object has `unit_ndims` equal to 2, since
    a single `Polygon` is represented by an array of shape `(n,d)`,
    where `n` is the number of vertices and `d` is the dimension of
    the underlying vector space. So, a `Polygon` object whose
    underlying array has shape `(5, 6, 4, 3)` represents a 5x6 array
    of quadrilaterals in RP^2 (i.e. the projectivization of R^3).

    """
    def __init__(self, proj_data, aux_data=None, dual_data=None,
                 unit_ndims=1, aux_ndims=0, dual_ndims=0):
        """Parameters
        -----------

        proj_data : ndarray
            underyling data describing this projective object

        aux_data : ndarray
            auxiliary data describing this projective
            object. Auxiliary data is any data which is in principle
            computable from `proj_data`, but is convenient to keep as
            part of the object definition for transformation purposes.

        dual_data : ndarray
            data describing this projective object which transforms
            covariantly, i.e. as a dual vector in projective space.

        unit_ndims : int
            number of ndims of an array representing a "unit" version
            of this object. For example, an object representing a
            single point in hyperbolic space has `unit_ndims` 1, while
            an object representing a line segment has `unit_ndims`
            equal to 2.

        aux_ndims : int
            like `unit_ndims`, but for auxiliary data.

        dual_ndims : int
            like `unit_ndims`, but for covariant (dual) data.

        """
        self.unit_ndims = unit_ndims
        self.aux_ndims = aux_ndims
        self.dual_ndims = dual_ndims

        try:
            self._construct_from_object(proj_data)
        except TypeError:
            self.set(proj_data, aux_data, dual_data)

    @property
    def dimension(self):
        return self.proj_data.shape[-1] - 1

    def _assert_geometry_valid(self, proj_data):
        if proj_data.ndim < self.unit_ndims:
            raise GeometryError(
                ("{} expects an array with ndim at least {}, got array of shape {}"
                ).format(
                    self.__class__.__name__, self.unit_ndims, proj_data.shape
                )
            )
    def _assert_aux_valid(self, aux_data):
        if aux_data is None and self.aux_ndims == 0:
            return

        if aux_data.ndim < self.aux_ndims:
            raise GeometryError(
                ("{} expects an auxiliary array with ndim at least {}, got array of shape"
                ).format(
                    self.__class__.__name__, self.unit_ndims, proj_data.shape
                )
            )

    def _compute_aux_data(self, proj_data):
        return None

    def _construct_from_object(self, hyp_obj):
        """if we're passed a hyperbolic object or an array of hyperbolic
        objects, build a new one out of them

        """

        try:
            self.set(hyp_obj.proj_data,
                     aux_data=hyp_obj.aux_data,
                     dual_data=hyp_obj.dual_data)
            return
        except AttributeError:
            pass

        try:
            unrolled_obj = list(hyp_obj)

            if len(unrolled_obj) == 0:
                raise IndexError

            hyp_array = np.array([obj.proj_data for obj in unrolled_obj])
            aux_array = np.array([obj.aux_data for obj in unrolled_obj])
            dual_array = np.array([obj.dual_data for obj in unrolled_obj])

            if (aux_array == None).any():
                aux_array = None

            if (dual_array == None).any():
                dual_array = None

            self.set(hyp_array, aux_data=aux_array,
                     dual_data=dual_array)
            return

        except (TypeError, AttributeError, IndexError) as e:
            pass

        raise TypeError

    def obj_shape(self):
        """Get the shape of the ndarray of "unit objects" this
        ProjectiveObject represents.

        Returns
        --------
        tuple


        """
        return self.proj_data.shape[:-1 * self.unit_ndims]

    def set(self, proj_data, aux_data=None, dual_data=None):
        """set the underlying data of the hyperbolic object.

        Subclasses may override this method to give special names to
        portions of the underlying data.

        Parameters
        -------------
        proj_data : ndarray
            underyling data representing this projective object.

        aux_data : ndarray
            underyling auxiliary data for this projective object.

        dual_data : ndarray
            underlying dual data for this projective object.

        """

        # TODO: assert dual geometry valid here as well.  Right now we
        # don't bother because the only dual geometry we're using is
        # technically also auxilliary...

        self._assert_geometry_valid(proj_data)
        if aux_data is None:
            aux_data = self._compute_aux_data(proj_data)

        self._assert_aux_valid(aux_data)

        self.proj_data = proj_data
        self.aux_data = aux_data
        self.dual_data = dual_data

    def flatten_to_unit(self, unit=None):
        """Get a flattened version of the projective object.

        This method reshapes the underlying data of the projective
        object to get a "flat" composite list of objects. For example,
        if called on a Segment object whose underlying array has shape
        (4, 5, 2, 3), this method uses the `unit_ndims` data member to
        interprets this array as an array of segments with shape
        (4,5), and returns a Segment object whose underlying array has
        shape (20, 2, 3).

        Parameters
        ----------

        unit : int
            the number of ndims to treat as a "unit" when flattening
            this object into units.

        """

        aux_unit = unit
        dual_unit = unit
        if unit is None:
            unit = self.unit_ndims
            aux_unit = self.aux_ndims
            dual_unit = self.dual_ndims

        flattened = copy(self)
        new_shape = (-1,) + self.proj_data.shape[-1 * unit:]
        new_proj_data = np.reshape(self.proj_data, new_shape)

        new_aux_data = None
        if self.aux_data is not None:
            new_aux_shape = (-1,) + self.aux_data.shape[-1 * aux_unit:]
            new_aux_data = np.reshape(self.aux_data, new_aux_shape)

        new_dual_data = None
        if self.dual_data is not None:
            new_dual_shape = (-1,) + self.dual_data.shape[-1 * dual_unit:]
            new_dual_data = np.reshape(self.dual_data, new_dual_shape)

        flattened.set(new_proj_data, aux_data=new_aux_data,
                      dual_data=new_dual_data)

        return flattened

    def flatten_to_aux(self):
        return self.flatten_to_unit(self.aux_ndims)

    def __repr__(self):
        return "({}, {})".format(
            self.__class__,
            self.proj_data.__repr__()
        )

    def __str__(self):
        return "{} with data:\n{}".format(
            self.__class__.__name__, self.proj_data.__str__()
        )

    def __getitem__(self, item):
        return self.__class__(self.proj_data[item])

    def projective_coords(self, proj_data=None):
        """Wrapper for ProjectiveObject.set, since underlying coordinates are
        projective."""
        if proj_data is not None:
            self.set(proj_data)

        return self.proj_data

    def affine_coords(self, aff_data=None, chart_index=0):
        """Get or set affine coordinates for this object.

        Parameters
        ----------
        aff_data : ndarray
            if not `None`, coordinate data for this point in an affine
            chart.

        chart_index : int
            index of standard affine chart to get/set coordinates in

        Returns
        -------
        ndarray
            affine coordinates of this Point, in the specified
            standard affine chart.

        """
        if aff_data is not None:
            self.set(projective_coords(aff_data, chart_index=chart_index))

        return affine_coords(self.proj_data, chart_index=chart_index,
                             column_vectors=False)

class Point(ProjectiveObject):
    """A point (or collection of points) in projective space.
    """

    def __init__(self, point, chart_index=None):
        """Parameters
        ----------
        point : ndarray or ProjectiveObject or iterable
            Data to use to construct a Point object. If `point` is an
            `ndarray`, then it is interpreted as data in either
            projective or affine coordinates, depending on whether
            `chart_index` is specified. If `point` is a
            `ProjectiveObject`, then construct a `Point` from the
            underlying data of that object.
        chart_index : int
            if `None` (default), then assume that `point` is data in
            projective coordinates, or a ProjectiveObject. Otherwise,
            interpret `point` as data in affine coordinates, and use
            `chart_index` as the index of the standard affine chart
            those coordinates are in.

        """

        self.unit_ndims = 1
        self.aux_ndims = 0
        self.dual_ndims = 0

        try:
            self._construct_from_object(point)
            return
        except TypeError:
            pass

        if chart_index is None:
            self.projective_coords(point)
        else:
            self.affine_coords(point, chart_index)

class PointPair(Point):
    """A pair of points (or a composite object consisting of a collection
    of pairs of points) in projective space.

    This is mostly useful as an interface for subclasses which provide
    more involved functionality.
    """
    def __init__(self, endpoint1, endpoint2=None):
        """If `endpoint2` is `None`, interpret `endpoint1` as either an
        `ndarray` of shape (2, ..., n) (where n is the dimension of
        the underlying vector space), or else a composite `Point`
        object which can be unpacked into two Points (which may
        themselves be composite).

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
        self.unit_ndims = 2
        self.aux_ndims = 0
        self.dual_ndims = 0

        if endpoint2 is None:
            try:
                self._construct_from_object(endpoint1)
                return
            except (AttributeError, TypeError, GeometryError):
                pass

        self.set_endpoints(endpoint1, endpoint2)

    def set(self, proj_data, **kwargs):
        ProjectiveObject.set(self, proj_data)
        self.endpoints = self.proj_data[..., :2, :]

    def set_endpoints(self, endpoint1, endpoint2=None):
        """Set the endpoints of a segment.

        If `endpoint2` is `None`, expect `endpoint1` to be an array of
        points with shape `(..., 2, n)`. Otherwise, expect `endpoint1`
        and `endpoint2` to be arrays of points with the same shape.

        Parameters
        ----------
        endpoint1 : Point or ndarray
            One (or both) endpoints of the point pair
        endpoint2 : Point or ndarray
            The other endpoint of the point pair. If `None`,
            `endpoint1` contains the data for both points in the pair.

        """
        if endpoint2 is None:
            self.set(Point(endpoint1).proj_data)
            return

        pt1 = Point(endpoint1)
        pt2 = Point(endpoint2)
        self.set(np.stack([pt1.proj_data, pt2.proj_data], axis=-2))

    def get_endpoints(self):
        """Get a Point representing the endpoints of this pair

        Returns
        --------
        Point
            A composite Point object representing the endpoints of
            this (possibly composite) PointPair

        """
        return Point(self.endpoints)

    def get_end_pair(self):
        """Return a pair of point objects, one for each endpoint

        Returns
        -----------
        tuple
            Tuple of the form `(endpoint1, endpoint2)`, where
            `endpoint1` and `endpoint2` are (possibly composite)
            `Point` objects representing the endpoints of this pair

        """
        return (Point(self.endpoints[..., 0, :]),
                Point(self.endpoints[..., 1, :]))

    def endpoint_affine_coords(self, chart_index=0):
        """Get endpoints of this segment in affine coordinates

        Parameters
        ----------
        chart_index : int
            Index of the standard affine chart to take coordinates in

        Returns
        --------
        ndarray
            Affine coordinates of the endpoints of this pair of
            points.

        """
        return self.get_endpoints().affine_coords(chart_index=chart_index)

    def endpoint_projective_coords(self):
        """Get endpoints of this segment in projective coordinates.

        Returns
        --------
        ndarray
            Projective coordinates of the endpoints of this pair of
            points.

        """
        return self.get_endpoints().projective_coords()

class Polygon(Point):
    """A finite-sided polygon in projective space.
    """
    def __init__(self, vertices, aux_data=None):
        """
        Parameters
        ----------
        vertices : Point or ndarray
            vertices of the polygon, as either an ndarray or a
            composite Point object (provided in the proper order for
            this polygon).

        aux_data : PointPair or ndarray
            Data to use to construct the edges of the polygon. If
            `None`, use the vertex data to compute the edge data.

        """
        ProjectiveObject.__init__(self, vertices, aux_data,
                                  unit_ndims=2, aux_ndims=3)

    def set(self, proj_data, aux_data=None, dual_data=None):
        ProjectiveObject.set(self, proj_data, aux_data)
        self.vertices = self.proj_data
        self.edges = self.aux_data

    def _compute_aux_data(self, proj_data):
        segments = PointPair(proj_data, np.roll(proj_data, -1, axis=-2))
        return segments.proj_data

    def get_edges(self):
        """Get the edges of this polygon

        Returns
        --------
        PointPair
            Edges of this polygon, as a composite PointPair object.

        """
        return PointPair(self.edges)

    def get_vertices(self):
        """Get the vertices of the polygon.

        Returns
        --------
        Point
            Vertices of this polygon, as a composite Point object.

        """
        return (self.proj_data)

class ConvexPolygon(Polygon):
    """A finite-sided convex polygon in projective space.
    """

    def __init__(self, vertices, aux_data=None, dual_data=None):
        r"""When providing point data for this polygon, non-extreme points
        (i.e. points in the interior of the convex hull) are
        discarded. To determine which points lie in the interior of
        the convex hull, the constructor either:

        - uses the provided `dual_data` to determine an affine
          chart in which the convex polygon lies (this chart is
          the complement of the hyperplane specified in
          `dual_data`), or

        - interprets the projective coordinates of the provided points
          as preferred lifts of those points in \(\mathbb{R}^n\), and
          computes an affine chart containing the projectivization of
          the convex hull of those lifts.

        Parameters
        ----------
        vertices : Point or ndarray
            points contained in this polygon
        aux_data : PointPair or None
            Data to construct the edges of this polygon. If `None`,
            use vertex data to construct edge data.
        dual_data : ProjectiveObject
            Dual vector specifying an affine chart containing every
            point in this convex polygon. If `None`, then compute a
            dual vector using lifts of the vertex data.

    """

        ProjectiveObject.__init__(self, vertices, aux_data=aux_data,
                                  dual_data=dual_data, unit_ndims=2,
                                  aux_ndims=3, dual_ndims=1)
    def add_points(self, points, in_place=False):
        """Add points to an existing convex polygon.

        Parameters
        ----------
        points : Point or ndarray
            Points to add to the convex polygon. Redundant points
            (lying in the interior of the convex hull) will be
            discarded.
        in_place : bool
            if `True`, modify this convex polygon object in-place
            instead of returning a new one.

        Raises
        ------
        GeometryError
            Raised if points are added to a composite ConvexPolygon
            (currently unsupported)

        Returns
        --------
        ConvexPolygon
            if `in_place` is `False`, return a modified ConvexPolygon
            object with the new points added.

        """

        if len(self.proj_data.shape) > 2:
            raise GeometryError(
                "Adding new points to a composite ConvexPolygon object is currently"
                " unsupported."
            )

        to_add = Point(points)
        new_data = np.concatenate((self.proj_data, to_add.proj_data), axis=-2)

        if in_place:
            self.set(new_data)
            return

        new_poly = ConvexPolygon(new_data)
        return new_poly

    def _convexify(self):
        if len(self.proj_data.shape) > 2:
            raise GeometryError(
                "Cannot auto-convexify a composite ConvexPolygon object."
            )

        dim = self.proj_data.shape[-1]
        self.dual_data = utils.find_positive_functional(self.proj_data)

        to_std_aff = np.linalg.inv(utils.find_isometry(np.identity(dim),
                                                       self.dual_data))

        standardized_coords = Point(
            self.proj_data @ to_std_aff
        ).affine_coords(chart_index=0)

        vertex_indices = ConvexHull(standardized_coords).vertices

        self.proj_data = self.proj_data[vertex_indices]
        self.aux_data = self._compute_aux_data(self.proj_data)

    def set(self, proj_data, aux_data=None, dual_data=None):
        ProjectiveObject.set(self, proj_data, aux_data, dual_data)
        if dual_data is None:
            self._convexify()

class Transformation(ProjectiveObject):
    """A projective transformation (or a composite object consisting of a
    collection of projective transformations).
    """
    def __init__(self, proj_data, column_vectors=False):
        """By default, the underlying data for a projective
        transformation is a *row matrix* (or an ndarray of row
        matrices), acting on vectors on the *right*.

        Parameters
        ----------
        proj_data : ProjectiveObject or ndarray
            Data to use to construct a projective transformation (or
            array of projective transformations).
        column_vectors : bool
            If `True`, interpret proj_data as a *column matrix* acting
            on the left. Otherwise proj_data gives a *row matrix*.
        """
        self.unit_ndims = 2
        self.aux_ndims = 0
        self.dual_ndims = 0

        try:
            self._construct_from_object(proj_data)
        except TypeError:
            if column_vectors:
                self.set(proj_data.swapaxes(-1,-2))
            else:
                self.set(proj_data)

    def _assert_geometry_valid(self, proj_data):
        ProjectiveObject._assert_geometry_valid(self, proj_data)
        if (len(proj_data.shape) < 2 or
            proj_data.shape[-2] != proj_data.shape[-1]):
            raise GeometryError(
                ("Projective transformation must be ndarray of n x n"
                 " matrices, got array with shape {}").format(
                     proj_data.shape))

    def set(self, proj_data, **kwargs):
        ProjectiveObject.set(self, proj_data)
        self.matrix = proj_data
        self.proj_data = proj_data

    def _apply_to_data(self, proj_data, broadcast, unit_ndims=1, dual=False):
        matrix = self.matrix
        if dual:
            matrix = np.linalg.inv(matrix).swapaxes(-1, -2)
        return utils.matrix_product(proj_data,
                                    matrix,
                                    unit_ndims, self.unit_ndims,
                                    broadcast=broadcast)

    def apply(self, proj_obj, broadcast="elementwise"):
        """Apply this transformation to another object in projective space.

        Parameters
        ----------
        proj_obj : ProjectiveObject or ndarray
            Projective object to apply this transformation to. This object
            may be composite.
        broadcast : string
            Broadcasting behavior for applying composite
            transformation objects. If "elementwise", then the shape
            of this (composite) transformation object and the shape of
            the (composite) object to apply transformations to need to
            be broadcastable. If "pairwise", then apply every element
            of this (composite) transformation object to every element
            of the target object (i.e. take an outer product).

        Returns
        -------
        ProjectiveObject
            Transformed (possibly composite) projective object. The
            type of this object is the same type as the original
            (untransformed) object. If the original object was
            provided as an ndarray, then the returned object has type
            ProjectiveObject.

        """
        new_obj = copy(proj_obj)

        try:
            proj_data = new_obj.proj_data
            proj_product = self._apply_to_data(new_obj.proj_data, broadcast,
                                               new_obj.unit_ndims)
            aux_data = new_obj.aux_data
            aux_product = None

            if aux_data is not None:
                aux_product = self._apply_to_data(new_obj.aux_data, broadcast,
                                                  new_obj.aux_ndims)

            dual_data = new_obj.dual_data
            dual_product = None
            if dual_data is not None:
                dual_product = self._apply_to_data(new_obj.dual_data, broadcast,
                                                   new_obj.dual_ndims)

            new_obj.set(proj_product,
                        aux_data=aux_product,
                        dual_data=dual_product)

            return new_obj
        except AttributeError:
            pass

        #otherwise, it's an array of vectors which we'll interpret as
        #some kind of hyperbolic object
        product = self._apply_to_data(proj_obj, broadcast)
        return self._data_to_object(product)

    def _data_to_object(self, data):
        return ProjectiveObject(data)

    def inv(self):
        """Get the inverse of this transformation.

        Returns
        --------
        ProjectiveTransformation
            Inverse of this transformation.
        """
        return self.__class__(np.linalg.inv(self.matrix))

    def __matmul__(self, other):
        return self.apply(other)

class ProjectiveRepresentation(representation.Representation):
    """A representation (of a free group) lying in PGL(V). Passing words
    (in the generators) to this representation yields `Transformation`
    objects.
    """
    def __getitem__(self, word):
        matrix = self._word_value(word)
        return Transformation(matrix, column_vectors=True)

    def __setitem__(self, generator, matrix):
        transform = Transformation(matrix, column_vectors=True)
        representation.Representation.__setitem__(self, generator,
                                                  transform.matrix.T)

    def transformations(self, words):
        """Get a composite transformation, representing a sequence of words in
        the generators for this representation.

        Parameters
        ----------
        words : iterable of strings
            Sequence of words to apply this representation to.

        Returns
        --------
        Transformation
            Composite transformation object containing one
            transformation for each word in `words`.

    """
        matrix_array = np.array(
            [representation.Representation.__getitem__(self, word)
             for word in words]
        )
        return Transformation(matrix_array, column_vectors=True)

    def automaton_accepted(self, automaton, length,
                           maxlen=True, with_words=False,
                           start_state=None, end_state=None,
                           precomputed=None):
        result = representation.Representation.automaton_accepted(
            self, automaton, length,
            with_words=with_words,
            start_state=start_state,
            end_state=end_state,
            precomputed=precomputed
        )

        if with_words:
            matrix_array, words = result
        else:
            matrix_array = result

        transformations = Transformation(matrix_array, column_vectors=True)

        if with_words:
            return transformations, words

        return transformations


def hyperplane_coordinate_transform(normal):
    r"""Find an orthogonal matrix taking the affine chart \(\{\vec{x} :
       \vec{x} \cdot \vec{n} \ne 0\}\) to the standard affine chart
       \(\{\vec{x} = (x_0, \ldots, x_n) : x_0 \ne 0\}\).

    Parameters
    ----------
    normal : array
        The vector \(\vec{n}\), normal to some hyperplane in R^n.

    Returns
    --------
    Transformation
        Projective transformation (orthogonal in the standard inner
        product on R^n) taking the desired affine chart to the
        standard chart with index 0.

    """
    mat = utils.find_definite_isometry(normal)
    return Transformation(mat, column_vectors=True).inv()

def affine_coords(points, chart_index=None, column_vectors=False):
    """Get affine coordinates for an array of points in projective space
    in one of the standard affine charts.

    Parameters
    -----------
    points: ndarray
        `ndarray` of points in projective space. the last dimension is
        assumed to be the same as the dimension of the underlying
        vector space.

    chart_index: int
        which of the n affine charts to take coordinates in. If
        `None`, determine the chart automatically.

    column_vectors: bool
        if `True`, interpret the second-to-last index as the dimension
        of the underlying vector space.

    Returns
    --------
    ndarray
        If chart_index is specified, return an array of points in
        affine coordinates in that chart. Otherwise, return a tuple
        `(affine, chart_index)`, where `chart_index` is the affine
        chart used.

        If `column_vectors` is `False` (the default), then the last
        index of the returned array is the dimension of the affine
        space. Otherwise, the second-to-last index is the dimension of
        affine space.

    """
    apoints = np.array(points)

    if column_vectors:
        apoints = apoints.swapaxes(-1, -2)

    _chart_index = chart_index

    #auto-determine chart
    if chart_index is None:
        _chart_index = np.argmax(
            np.min(np.abs(apoints), axis=tuple(range(len(apoints.shape) - 1)))
        )

    if (apoints[..., _chart_index] == 0).any():
        if chart_index is not None:
            raise GeometryError(
                "points don't lie in the specified affine chart"
            )
        else:
            raise GeometryError(
                "points don't lie in any standard affine chart"
            )

    affine = np.delete(
        (apoints.T / apoints.T[_chart_index]).T,
        _chart_index, axis=-1
    )

    if column_vectors:
        affine = affine.swapaxes(-1, -2)

    if chart_index is None:
        return (affine, _chart_index)
    else:
        return affine

def projective_coords(points, chart_index=0, column_vectors=False):
    """Get projective coordinates for points in affine space

    Parameters
    ----------
    points : ndarray or sequence
        Points in affine coordinates. The last dimension of the array
        is the dimension of affine space.

    chart_index: int
        Index of the affine chart we assume these points are lying in

    column_vectors: bool
        If `True`, interpret the second-to-last index as the dimension
        of affine space.

    Returns
    --------
    ndarray
        Projective coordinates of the given points. The last dimension
        of the array is the dimension of the underlying vector space
        (or the second-to-last dimension, if `column_vectors` is
        `True`).

    """
    coords = np.array(points)

    if column_vectors:
        coords = coords.swapaxes(-1, -2)

    result = np.zeros(coords.shape[:-1] + (coords.shape[-1] + 1,))
    indices = np.arange(coords.shape[-1])
    indices[chart_index:] += 1

    result[..., indices] = coords
    result[..., chart_index] = 1.

    if column_vectors:
        result = result.swapaxes(-1, -2)

    return result

def identity(dimension):
    """Get the identity projective transformation.

    Parameters
    ----------
    dimension : int
        Dimension of the projective space to act on.

    Returns
    --------
    Transformation
        The identity map on RP^n, where n = `dimension`.

    """

    return Transformation(np.identity(dimension + 1))

def affine_linear_map(linear_map, chart_index=0, column_vectors=True):
    """Get a projective transformation restricting to a linear map on a
       standard affine chart.

    Parameters
    ----------
    linear_map : ndarray
        A linear map giving an affine transformation on a standard
        affine chart

    chart_index : int
        Index of the standard affine chart where this projective
        transformation acts

    column_vectors :
        It `True`, interpret `linear_map` as a matrix acting on column
        vectors (on the left). Otherwise, `linear_map` acts on row
        vectors (on the right).

    Returns
    --------
    Transformation
        Projective transformation preserving a standard affine chart
        and acting by a linear map on that affine space (i.e. fixing a
        point in that affine space).

    """
    h, w = linear_map.shape

    tf_mat = np.block(
        [[linear_map[:chart_index, :chart_index],
          np.zeros((chart_index, 1)), linear_map[:chart_index, chart_index:]],
         [np.zeros((1, chart_index)), 1., np.zeros((1, w - chart_index))],
         [linear_map[chart_index:, :chart_index],
          np.zeros((h - chart_index, 1)), linear_map[chart_index:, chart_index:]]])

    return Transformation(tf_mat, column_vectors=column_vectors)

def affine_translation(translation, chart_index=0):
    """Get a translation in a standard affine chart.

    Parameters
    ----------
    translation : ndarray
        vector to translate along in affine space
    chart_index : int
        index of the standard affine chart this translation acts on

    Returns
    --------
    Transformation
        Projective transformation preserving a standard affine chart
        and acting by an affine translation in that affine space.

    """
    tf = np.identity(len(translation) + 1)
    tf[chart_index, :chart_index] = translation[:chart_index]
    tf[chart_index, chart_index + 1:] = translation[chart_index:]

    return Transformation(tf, column_vectors=False)
