import numpy as np
from scipy.optimize import fsolve

from geometry_tools import projective


def bilinear_form(matrix, v1, v2):
    w, h = matrix.shape
    v1 = v1.reshape((1, w))
    v2 = v2.reshape((h, 1))

    return (v1 @ matrix @ v2)[0,0]

def hyp_to_affine_dist(r):
    return (np.exp(2 * r) - 1) / (1 + np.exp(2 * r))

def rotation_matrix(angle):
    return np.array([[np.cos(angle), -1*np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])

def block_diagonal(matrices):
    mats = [np.matrix(matrix) for matrix in matrices]
    n = len(mats)
    arr = [[None] * n for i in range(n)]
    for i, matrix_i in enumerate(mats):
        for j, matrix_j in enumerate(mats):
            if i == j:
                arr[i][j] = matrix_i
            else:
                h, _ = matrix_i.shape
                _, w = matrix_j.shape
                arr[i][j] = np.zeros((h, w))
    return np.block(arr)

class HyperbolicPoint:
    def __init__(self, space, point, coords="kleinian"):
        self.space = space
        try:
            self.hyperboloid_coords(point.coords)
        except AttributeError:
            if coords == "kleinian":
                self.kleinian_coords(point)
            elif coords == "projective":
                self.projective_coords(point)

    def kleinian_coords(self, vector=None):
        if vector is not None:
            projective = self.space.projective.affine_to_projective([vector])
            self.projective_coords(projective)

        return self.space.projective.affine_coordinates([self.coords])

    def projective_coords(self, vector=None):
        if vector is not None:
            self.coords = self.space.hyperboloid_coords(vector)

        return self.coords

    def hyperboloid_coords(self, vector=None):
        if vector is not None:
            self.coords = self.space.hyperboloid_coords(vector)

        return self.coords

    def distance(self, other):
        product = bilinear_form(self.space.minkowski(),
                                self.coords, other.coords)
        return np.arccosh(np.abs(product))

    def unit_tangent_towards(self, other):
        diff = other.coords - self.coords
        tv = HyperbolicTangentVector(self.space, self, diff)
        tv.normalize()
        return tv

    def translation_to_origin(self):
        klein_vec = self.kleinian_coords()
        block = projective.orthogonal_transform(klein_vec)

        elliptic = self.space.get_elliptic(block)

        affine_coords = elliptic.apply(self).kleinian_coords().flatten()
        aff_param = affine_coords[0]

        param = np.sqrt((1 - aff_param) / (1 + aff_param))

        return elliptic.inv() @ self.space.get_standard_loxodromic(param) @ elliptic

class HyperbolicTangentVector:
    def __init__(self, space, point, vector):
        self.point = HyperbolicPoint(space, point)
        self.space = space
        self.vector = self.space.project_to_hyperboloid(
            self.point.coords,
            vector
        )

    def normalize(self):
        norm2 = bilinear_form(self.space.minkowski(), self.vector, self.vector)
        if norm2 != 0.0:
            self.vector = self.vector / np.abs(np.sqrt(norm2))

    def translation_to_origin(self):
        point_translation = self.point.translation_to_origin()
        translated = point_translation.apply(self)
        affine_tangent_vector = translated.vector[1:]
        block_elliptic = projective.orthogonal_transform(affine_tangent_vector)

        elliptic = self.space.get_elliptic(block_elliptic)

        return elliptic @ point_translation

    def point_along(self, distance):
        affine_point = np.concatenate([
            [hyp_to_affine_dist(distance)],
            np.zeros(self.space.dimension - 1)
        ])
        basept = self.space.get_point(affine_point, coords="kleinian")

        return self.translation_to_origin().inv().apply(basept)

    def angle(self, other):
        return self.space.hyperbolic_angle(self.point.coords,
                                           self.vector,
                                           other.vector)

class HyperbolicPoints:
    def __init__(self, space, points, type="pointlist"):
        self.space = space
        if type == "pointlist":
            self.points = np.column_stack(
                [HyperbolicPoint(self.space, point).coords for point in points]
            )
        else:
            self.points = points

    def kleinian_coordinates(self):
        return self.space.projective.affine_coordinates(self.points, type=None)

    def kleinian_xy_coordinates(self):
        klein_pts = self.kleinian_coordinates()
        return (klein_pts[:, 0], klein_pts[:, 1])

    def __getitem__(self, n):
        return HyperbolicPoint(self.space, self.points[:, n],
                               coords="projective")

class HyperbolicIsometry:
    def __init__(self, space, matrix):
        self.space = space
        self.matrix = matrix #in PO(d, 1)

    def apply(self, hyp_obj):
        try:
            point, vector = hyp_obj.point, hyp_obj.vector
            return HyperbolicTangentVector(self.space,
                                           self.apply(point),
                                           self.matrix @ vector)
        except AttributeError:
            pass
        try:
            return HyperbolicPoint(self.space,
                                   self.matrix @ hyp_obj.coords,
                                   coords="projective")
        except AttributeError:
            pass

        return HyperbolicPoints(self.space,
                                self.matrix @ hyp_obj.points,
                                type="pointlist")


    def inv(self):
        return HyperbolicIsometry(self.space, np.linalg.inv(self.matrix))

    def __matmul__(self, other):
        return HyperbolicIsometry(self.space, self.matrix @ other.matrix)


class HyperbolicSpace:
    """Class to work with the projective model for hyperbolic space,
    coordinatized as the unit circle in a standard affine chart.

    """
    def __init__(self, dimension):
        self.projective = projective.ProjectiveSpace(dimension + 1)
        self.dimension = dimension

    def minkowski(self):
        return np.diag(np.concatenate(([-1.0], np.ones(self.dimension))))

    def hyperboloid_coords(self, vector):
        r_vec = vector.reshape((self.dimension + 1, 1))
        norm2 = np.abs(bilinear_form(self.minkowski(), r_vec, r_vec))
        return vector / np.sqrt(norm2)

    def get_elliptic(self, block_elliptic):
        mat = block_diagonal([1.0, block_elliptic])

        return HyperbolicIsometry(self, mat)

    def hyperboloid_projection(self, basepoint):
        basepoint = basepoint.reshape((self.dimension + 1, 1))
        normal = self.minkowski() @ basepoint
        unit_normal = normal / np.linalg.norm(normal)

        return np.identity(self.dimension + 1) - (unit_normal @ unit_normal.reshape((1, self.dimension + 1)))

    def project_to_hyperboloid(self, basepoint, tangent_vector):
        projection = self.hyperboloid_projection(basepoint)
        tangent_vector = tangent_vector.reshape((self.dimension + 1, 1))
        return projection @ tangent_vector

    def riemannian_metric(self, point):
        projection = self.hyperboloid_projection(point)
        return projection.T @ self.minkowski() @ projection

    def hyperbolic_angle(self, point, v1, v2):
        metric = self.riemannian_metric(point)
        prod = bilinear_form(metric, v1, v2)
        norm1 = bilinear_form(metric, v1, v1)
        norm2 = bilinear_form(metric, v2, v2)

        return np.arccos(prod / np.abs(np.sqrt(norm1 * norm2)))

    def get_origin(self):
        return self.get_point([0., 0.])

    def get_point(self, point, coords="kleinian"):
        return HyperbolicPoint(self, point, coords)

    def loxodromic_basis_change(self):
        basis_change = np.matrix([
            [1.0, 1.0],
            [1.0, -1.0]
        ])
        return block_diagonal([basis_change, np.identity(self.dimension - 1)])

    def get_standard_loxodromic(self, parameter):
        basis_change = self.loxodromic_basis_change()
        diagonal_loxodromic = np.diag(
            np.concatenate(([parameter, 1.0/parameter],
                       np.ones(self.dimension - 1)))
        )

        return HyperbolicIsometry(self,
            basis_change @ diagonal_loxodromic @ np.linalg.inv(basis_change)
        )

    def get_standard_rotation(self, angle):
        affine = block_diagonal([rotation_matrix(angle),
                                 np.identity(self.dimension - 2)])
        return self.get_elliptic(affine)

    def regular_polygon(self, n, hyp_radius):
        origin = self.get_point([0., 0.])
        tangent = HyperbolicTangentVector(self, origin, np.array([0.0, 1.0, 0.0]))
        tangent.normalize()

        start_vertex = tangent.point_along(hyp_radius)
        rotation = self.get_standard_rotation(2 * np.pi / n)

        vertices = [start_vertex]
        for i in range(n - 1):
            vertices.append(rotation.apply(vertices[-1]))

        return HyperbolicPoints(self, vertices)

    def polygon_interior_angle(self, n, hyp_radius):
        polygon = self.regular_polygon(n, hyp_radius)
        tv1 = polygon[0].unit_tangent_towards(polygon[-1])
        tv2 = polygon[0].unit_tangent_towards(polygon[1])

        return tv1.angle(tv2)

class HyperbolicPlane(HyperbolicSpace):
    def __init__(self):
        self.projective = projective.ProjectivePlane()
        self.dimension = 2

    def get_boundary_point(self, theta):
        return np.array([1.0, np.cos(theta), np.sin(theta)])

    def basis_change(self):
        basis_change = np.matrix([
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        return basis_change

    def center_point(self):
        return np.array([1.0, 0.0, 0.0]).T

    def get_standard_reflection(self):
        basis_change = self.basis_change
        diag_reflection = np.matrix([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        return basis_change @ diag_reflection @ np.linalg.inv(basis_change)
