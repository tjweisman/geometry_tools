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
    def __init__(self, proj_data, aux_data=None, dual_data=None,
                 unit_ndims=1, aux_ndims=0, dual_ndims=0):
        """Parameters
        -----------

        proj_data : ndarray
            underyling data describing this hyperbolic object

        aux_data : ndarray
            auxiliary data describing this hyperbolic
            object. Auxiliary data is any data which is in principle
            computable from `proj_data`, but is convenient to keep as
            part of the object definition for transformation purposes.

        dual_data : ndarray
            data describing this hyperbolic object which transforms
            covariantly, i.e. as a dual vector in projective space.

        unit_ndims : int
            number of ndims of an array representing a "unit" version
            of this object. For example, an object representing a
            single point in hyperbolic space has `unit_ndims` 1, while
            an object representing a line segment has `unit_ndims`
            equal to 2.

        aux_ndims : int
            like `unit_ndims`, but for auxiliary data.

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
        HyperbolicObject represents.

        Returns
        ---------
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
        """return a flattened version of the hyperbolic object.

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
        """wrapper for ProjectiveObject.set, since underlying coordinates are
        projective."""
        if proj_data is not None:
            self.set(proj_data)

        return self.proj_data

    def affine_coords(self, aff_data=None, chart_index=0):
        """Get or set affine coordinates for this object.
        """
        if aff_data is not None:
            self.set(projective_coords(aff_data, chart_index=chart_index))

        return affine_coords(self.proj_data, chart_index=chart_index,
                             column_vectors=False)

class Point(ProjectiveObject):
    def __init__(self, point, chart_index=None):
        self.unit_ndims = 1
        self.aux_ndims = 0

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
    def __init__(self, endpoint1, endpoint2=None):
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
        self.unit_ndims = 2
        self.aux_ndims = 0

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

        """
        if endpoint2 is None:
            self.set(Point(endpoint1).proj_data)
            return

        pt1 = Point(endpoint1)
        pt2 = Point(endpoint2)
        self.set(np.stack([pt1.proj_data, pt2.proj_data], axis=-2))

    def get_endpoints(self):
        return Point(self.endpoints)

    def get_end_pair(self, as_points=False):
        """Return a pair of point objects, one for each endpoint
        """
        if as_points:
            p1, p2 = self.get_end_pair(as_points=False)
            return (Point(p1), Point(p2))

        return (self.endpoints[..., 0, :], self.endpoints[..., 1, :])

    def endpoint_affine_coords(self, chart_index=0):
        return self.get_endpoints().affine_coords(chart_index=chart_index)

    def endpoint_projective_coords(self):
        return self.get_endpoints().projective_coords()

class Polygon(Point):
    def __init__(self, vertices, aux_data=None):
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
        return PointPair(self.edges)

    def get_vertices(self):
        return (self.proj_data)

class ConvexPolygon(Polygon):
    def __init__(self, vertices, aux_data=None, dual_data=None):
        ProjectiveObject.__init__(self, vertices, aux_data=aux_data,
                                  dual_data=dual_data, unit_ndims=2,
                                  aux_ndims=3, dual_ndims=1)
    def add_points(self, points, in_place=False):
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
    def __init__(self, proj_data, column_vectors=False):
        """Constructor for projective transformation.

        Underlying data is stored as row vectors, but by default the
        constructor accepts matrices acting on columns, since that's
        how my head thinks.

        """
        self.unit_ndims = 2
        self.aux_ndims = 0

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
        """Apply this isometry to another object in hyperbolic space.

        Broadcast is either "elementwise" or "pairwise", treating self
        and hyp_obj as ndarrays of isometries and hyperbolic objects,
        respectively.

        hyp_obj may be either a HyperbolicObject instance or the
        underlying data of one. In either case, this function returns
        a HyperbolicObject (of the same subclass as hyp_obj, if
        applicable).

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
        """invert the isometry"""
        return self.__class__(np.linalg.inv(self.matrix))

    def __matmul__(self, other):
        return self.apply(other)

class ProjectiveRepresentation(representation.Representation):
    def __init__(self, generator_names=[], normalization_step=-1):
        representation.Representation.__init__(self, generator_names,
                                               normalization_step)
    def __getitem__(self, word):
        matrix = self._word_value(word)
        return Transformation(matrix, column_vectors=True)

    def __setitem__(self, generator, isometry):
        try:
            super().__setitem__(generator, isometry.matrix.T)
            return
        except AttributeError:
            super().__setitem__(generator, isometry)

    @classmethod
    def from_matrix_rep(cls, rep, **kwargs):
        """Construct a hyperbolic representation from a matrix
        representation"""
        hyp_rep = cls(**kwargs)
        for g, matrix in rep.generators.items():
            hyp_rep[g] = matrix

        return hyp_rep

    def transformations(self, words):
        """Get an Isometry object holding the matrices which are the images of
        a sequence of words in the generators.

        """
        matrix_array = np.array(
            [representation.Representation.__getitem__(self, word)
             for word in words]
        )
        return Transformation(matrix_array, column_vectors=True)


def hyperplane_coordinate_transform(normal):
    """find an orthogonal matrix taking the the affine chart "x . normal
    != 0" to the standard affine chart "x_0 != 0

    """

    return np.linalg.inv(orthogonal_transform(normal)).T

def orthogonal_transform(v1, v2=None):
    """find an orthogonal matrix taking v1 to v2. If v2 is not specified,
    take v1 to the first standard basis vector (1, 0, ... 0).

    """
    if v2 is not None:
        orth1 = orthogonal_transform(v1)
        return np.matmul(orth1, np.linalg.inv(orthogonal_transform(v2)))
    else:
        Q, R = np.linalg.qr(
            np.column_stack([v1.reshape((v1.size, 1)),
                       np.identity(v1.size)])
        )
        if R[0,0] == 0:
            return np.identity(v1.size)

        sign = R[0,0] / np.abs(R[0,0])
        return np.linalg.inv(Q * sign)

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
    ndarray:
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
    ndarray:
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
    return Transformation(np.identity(dimension + 1))

class ProjectiveSpace:
    """class to model a copy of projective space.

    The most important functions are affine_coordinates and
    affine_to_projective, which respectively convert between affine
    and projective coordinates.

    The class also keeps track of a single coordinate transformation
    matrix, which is applied automatically when determining
    affine/projective coordinates.

    """
    def __init__(self, dim):
        self.dim = dim
        self.coordinates = np.identity(dim)

    def set_hyperplane_coordinates(self, normal):
        """set the coordinate transform to put a specified hyperplane at
        infinity.

        The hyperplane is of the form x . normal = 0, where . is the
        standard Euclidean inner product on R^n.
        """
        self.coordinates = hyperplane_coordinate_transform(normal)

    def postcompose_coordinates(self, coordinates):
        """postcompose coordinate transformation with a matrix"""
        self.coordinates = np.matmul(coordinates, self.coordinates)

    def precompose_coordinates(self, coordinates):
        """precompose coordinate transformation with a given matrix"""
        self.coordinates = np.matmul(self.coordinates, coordinates)

    def add_linear_transform(self, affine_transform):
        """postcompose the coordinate transformation with an affine map"""
        self.postcompose_coordinates(
            np.block([[1, np.zeros((1, self.dim - 1))],
                      [np.zeros((self.dim - 1, 1)), affine_transform]])
        )

    def add_affine_translation(self, translation):
        """postcompose the coordinate transformation with an affine
translation"""
        translation_col = translation.reshape((self.dim - 1,1))
        self.postcompose_coordinates(
            np.block([[1, np.zeros((1, self.dim - 1))],
                     [translation_col, np.identity(self.dim - 1)]])
        )

    def set_affine_origin(self, origin_vector):
        """postcompose coordinates with an affine translation which sets the
        origin to the given point"""
        s_vector = np.array(origin_vector).reshape((self.dim, 1))
        loc = affine_coords(
            np.matmul(self.coordinates, s_vector).T, 0).reshape((self.dim - 1,1))
        self.add_affine_translation(-1 * loc)

    def set_affine_direction(self, direction_vector, direction):
        """postcompose coordinates with a Euclidean rotation to get a
        direction vector pointing in a given direction.

        """
        s_vector = np.array(direction_vector).reshape((self.dim,1))
        d_vector = np.array(direction).reshape((self.dim - 1,1))

        loc = affine_coords(
            np.matmul(self.coordinates, s_vector).T, 0).reshape((self.dim - 1,1))
        transform = orthogonal_transform(loc, d_vector)
        self.add_linear_transform(transform)

    def set_coordinate_matrix(self, coordinate_matrix):
        self.coordinates = coordinate_matrix

    def affine_coordinates(self, point_list, type="rows", chart=0):
        """return affine coordinates of a set of vectors in R^(n+1)

        If type = "rows," then point_list is an ndarray where the last
        index is the dimension of underlying vector space. If type =
        "columns," then the second-to-last index is the dimension of
        the underlying vector space.

        """
        coords = np.array(point_list)
        if type == "rows":
            #last index is the dimension
            pts = coords @ self.coordinates.T
            return affine_coords(pts, chart)
        elif type == "columns":
            #second-to-last index is the dimension
            pts = (self.coordinates @ coords).swapaxes(-1, -2)
            return affine_coords(pts, chart).swapaxes(-1, -2)

        #guess the dimension axis from the array
        pts, dim_index = utils.dimension_to_axis(coords, self.dim, -1)
        pts = pts @ self.coordinates.T
        return affine_coords(pts, chart).swapaxes(dim_index, -1)

    def affine_to_projective(self, point_list, column_vectors=True):
        """return standard lifts of points in affine coordinates to R^(n+1).

        If type = "rows," then point_list is an ndarray where the last
        index is the dimension of the affine space. If type =
        "columns," then the second-to-last index is the dimension of
        the affine space.

        """
        coords = np.array(point_list)

        one_shape = list(coords.shape)
        if type == "rows":
            dim_index = -1
        elif type == "columns":
            dim_index = -2

        one_shape[dim_index] = 1
        ones = np.ones(one_shape)

        projective = np.concatenate([ones, coords], axis=dim_index)

        if type == "rows":
            return projective @ self.coordinates.T
        elif type == "columns":
            coords = self.coordinates @ projective.swapaxes(-1, -2)
            return coords.swapaxes(-1, -2)

class ProjectivePlane(ProjectiveSpace):
    """Projective space of dimension 2"""
    def __init__(self):
        super().__init__(3)

    def xy_coords(self, points):
        affine_pts = self.affine_coordinates(points)
        return affine_pts[:, 0], affine_pts[:, 1]
