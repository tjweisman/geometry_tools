"""This module provides classes and helper functions to work with
projective space in numerical coordinates.

The main class provided by this module is ProjectiveSpace, which
represents a copy of n-dimensional projective space.

"""

import numpy as np

class ProjectivizationException(Exception):
    pass

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

def affine_coords(points, chart_index=None):
    """get affine coordinates for an array of points in projective space
    in one of the standard affine charts.

    points: ndarray of points in projective space. the last dimension
    is assumed to be the same as the dimension of the underlying
    vector space.

    chart_index: which of the n affine charts to take coordinates
    in. If None, determine the chart automatically.

    Return value: If chart_index is specified, return an array of
    points in affine coordinates in that chart. Otherwise, return a
    tuple (affine, chart_index), where chart_index is the affine chart
    used.

    """
    apoints = np.array(points)

    _chart_index = chart_index

    #auto-determine chart
    if chart_index is None:
        _chart_index = np.argmax(
            np.min(np.abs(apoints), axis=tuple(range(len(apoints.shape) - 1)))
        )

    if (apoints.T[_chart_index] == 0).any():
        if chart_index is not None:
            raise ProjectivizationException(
                "points don't lie in the specified affine chart"
            )
        else:
            raise ProjectivizationException(
                "points don't lie in any standard affine chart"
            )

    affine = np.delete(
        (apoints.T / apoints.T[_chart_index]).T,
        _chart_index, axis=-1
    )

    if chart_index is None:
        return (affine, _chart_index)
    else:
        return affine

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

    def affine_to_projective(self, point_list, type="rows"):
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
