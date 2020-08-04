import numpy as np

class ProjectivizationException(Exception):
    pass

def hyperplane_coordinate_transform(normal):
    """find an matrix taking the the affine chart "x . normal != 0" to the
    standard affine chart "x_0 != 0"""

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
        return np.linalg.inv(Q)

def affine_coords(points, chart_index = None):
    apoints = np.array(points)

    _chart_index = chart_index
    if chart_index is None:
        _chart_index = np.argmax(np.amin(np.absolute(points), axis=0))

    if (apoints[:,_chart_index] == 0).any():
        if chart_index is not None:
            raise ProjectivizationException(
                "points don't lie in the specified affine chart"
            )
        else:
            raise ProjectivizationException(
                "points don't lie in any standard affine chart"
            )

    normalized = (apoints / apoints[:,_chart_index][:,None]).T

    affine = np.concatenate(
        [normalized[:_chart_index], normalized[_chart_index + 1:]]).T


    if chart_index is None:
        return (affine, _chart_index)
    else:
        return affine

class ProjectiveSpace:
    def __init__(self, dim):
        self.dim = dim
        self.coordinates = np.identity(dim)

    def set_hyperplane_coordinates(self, normal):
        self.coordinates = hyperplane_coordinate_transform(normal)

    def postcompose_coordinates(self, coordinates):
        self.coordinates = np.matmul(coordinates, self.coordinates)

    def precompose_coordinates(self, coordinates):
        self.coordinates = np.matmul(self.coordinates, coordinates)

    def add_linear_transform(self, affine_transform):
        self.postcompose_coordinates(
            np.block([[1, np.zeros((1, self.dim - 1))],
                      [np.zeros((self.dim - 1, 1)), affine_transform]])
        )

    def add_affine_translation(self, translation):
        translation_col = translation.reshape((self.dim - 1,1))
        self.postcompose_coordinates(
            np.block([[1, np.zeros((1, self.dim - 1))],
                     [translation_col, np.identity(self.dim - 1)]])
        )

    def set_affine_origin(self, origin_vector):
        s_vector = np.array(origin_vector).reshape((self.dim, 1))
        loc = affine_coords(
            np.matmul(self.coordinates, s_vector).T, 0).reshape((self.dim - 1,1))
        self.add_affine_translation(-1 * loc)

    def set_affine_direction(self, direction_vector, direction):
        s_vector = np.array(direction_vector).reshape((self.dim,1))
        d_vector = np.array(direction).reshape((self.dim - 1,1))

        loc = affine_coords(
            np.matmul(self.coordinates, s_vector).T, 0).reshape((self.dim - 1,1))
        transform = orthogonal_transform(loc, d_vector)
        self.add_linear_transform(transform)

    def set_coordinate_matrix(self, coordinate_matrix):
        self.coordinates = coordinates

    def affine_coordinates(self, point_list):
        pts = (np.matmul(self.coordinates, np.column_stack(point_list))).T
        return affine_coords(pts, 0)

    def affine_to_projective(self, point_list):
        pts = np.column_stack(point_list)
        _, num_pts = pts.shape
        return np.matmul(np.linalg.inv(self.coordinates),
                         np.block([[np.ones((1, num_pts))],
                                   [pts]]))


class ProjectivePlane(ProjectiveSpace):
    def __init__(self):
        super().__init__(3)

    def xy_coords(self, points):
        affine_pts = self.affine_coordinates(points)
        return affine_pts[:, 0], affine_pts[:, 1]


def lift(points, chart_index):
    return np.insert(points, chart_index, np.ones(points.shape[0]), axis=1)

def change_chart(points, chart_index, new_index):
    return affine_coords(lift(points, chart_index), new_index)

def sphere_lift(points):
    return lift(*affine_coords(points))

def guess_basis(points):
    dim = points.shape[1]
    center = np.mean(points, axis=0)
    center = center / np.linalg.norm(center)

    basis, inverse = (
        np.linalg.qr(np.column_stack((center, np.identity(dim))))
    )

    return (basis, inverse[:,1:])
