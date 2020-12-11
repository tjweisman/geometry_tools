from copy import copy

import numpy as np
import scipy
from scipy.optimize import fsolve

from geometry_tools import projective, representation, utils

#TODO: this is garbage
def bilinear_form(matrix, v1, v2):
    w, h = matrix.shape
    v1 = v1.reshape((1, w))
    v2 = v2.reshape((h, 1))

    return (v1 @ matrix @ v2)[0,0]

def hyp_to_affine_dist(r):
    return (np.exp(2 * r) - 1) / (1 + np.exp(2 * r))

class GeometryException(Exception):
    pass

class HyperbolicObject:
    def __init__(self, space, hyp_data):
        try:
            self._construct_from_object(hyp_data)
        except TypeError:
            self.space = space
            self.set(hyp_data)

    def _construct_from_object(self, hyp_obj):
        #if we're passed a hyperbolic object or an array of hyperbolic objects,
        #build a new one out of them
        try:
            self.space = hyp_obj.space
            self.set(hyp_obj.hyp_data)
            return
        except AttributeError:
            pass

        try:
            array = np.array([obj.hyp_data for obj in hyp_obj])
            self.set(array)
            return
        except (TypeError, AttributeError):
            pass

        raise TypeError

    def set(self, hyp_data):
        self.hyp_data = hyp_data

    def __repr__(self):
        return "('hyperbolic object', {})".format(
            self.hyp_data.__repr__()
        )

    def __str__(self):
        return "Hyperbolic object with data:\n" + self.hyp_data.__str__()

    def projective_coords(self, hyp_data=None):
        #all hyperbolic object data is stored as projective
        #coordinates
        if hyp_data is not None:
            self.set(hyp_data)

        return self.hyp_data

    def kleinian_coords(self, hyp_data=None):
        if hyp_data is not None:
            self.set(self.space.projective.affine_to_projective(hyp_data))

        return self.space.kleinian_coords(self.hyp_data)

class Point(HyperbolicObject):
    def __init__(self, space, point, coords="kleinian"):
        try:
            self._construct_from_object(point)
        except TypeError:
            self.space = space
            hyp_data = point
            if coords == "kleinian":
                self.kleinian_coords(point)
            else:
                self.set(point)

    def hyperboloid_coords(self, hyp_data=None):
        if hyp_data is not None:
            self.set(hyp_data)

        return self.space.hyperboloid_coords(self.hyp_data)

    def poincare_coords(self, hyp_data=None):
        if hyp_data is not None:
            klein = self.space.poincare_to_kleinian(hyp_data)
            self.kleinian_coords(klein)

        return self.space.kleinian_to_poincare(self.kleinian_coords())

    #TODO: rewrite these so they make sense for many points
    def distance(self, other):
        product = bilinear_form(self.space.minkowski(),
                                self.hyp_data, other.hyp_data)
        return np.arccosh(np.abs(product))

    def unit_tangent_towards(self, other):
        diff = other.hyp_data - self.hyp_data
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

    def collection(space, collection_data):
        return HyperbolicPoints(space, collection_data)

class IdealPoint(HyperbolicObject):
    pass

class Subspace(HyperbolicObject):
    #TODO: make a constructor that computes an orthogonal ideal basis
    #if we're not given one
    def set(self, hyp_data):
        HyperbolicObject.set(self, hyp_data)


        self.ideal_basis = hyp_data

    def ideal_basis_coords(self, coords="kleinian"):
        if coords == "kleinian":
            return self.space.kleinian_coords(self.ideal_basis)

        return self.ideal_basis

    def sphere_parameter(self):
        klein = self.ideal_basis_coords()

        poincare_midpoint = self.space.kleinian_to_poincare(
            klein.sum(axis=-2) / klein.shape[-2]
        )

        poincare_extreme = utils.sphere_inversion(poincare_midpoint)

        center = (poincare_midpoint + poincare_extreme) / 2
        radius = np.sqrt(utils.normsq(poincare_midpoint - poincare_extreme)) / 2

        return center, radius

    def circle_parameter(self):
        center, radius = self.sphere_parameter()
        klein = self.ideal_basis_coords()
        thetas = utils.circle_angles(center, klein)

        return center, radius, thetas

class Segment(Point, Subspace):
    def __init__(self, space, segment):
        try:
            self._construct_from_object(segment)
            self.endpoints = segment.endpoints
            self.ideal_basis = segment.ideal_basis
            return
        except (AttributeError, TypeError):
            pass

        self.space = space

        try:
            self.endpoints = segment.hyp_data
        except AttributeError:
            self.endpoints = segment

        self.compute_ideal_endpoints(self.endpoints)

    def from_endpoints(endpoints):
        return Segment(space, endpoints.hyp_data)

    def set(self, hyp_data):
        self.hyp_data = hyp_data
        self.endpoints = self.hyp_data[...,0,:,:]
        self.ideal_basis = self.hyp_data[...,1,:,:]

    def compute_ideal_endpoints(self, endpoints):
        products = endpoints @ self.space.minkowski() @ endpoints.swapaxes(-1, -2)
        a11 = products[..., 0, 0]
        a22 = products[..., 1, 1]
        a12 = products[..., 0, 1]

        a = a22
        b = 2 * a12
        c = a11

        mu1 = (-b + np.sqrt(b * b - 4 * a * c)) / (2*a)
        mu2 = (-b - np.sqrt(b * b - 4 * a * c)) / (2*a)

        null1 = endpoints[..., 0, :] + (mu1.T * endpoints[..., 1, :].T).T
        null2 = endpoints[..., 0, :] + (mu2.T * endpoints[..., 1, :].T).T

        self.ideal_basis = np.array([null1, null2]).swapaxes(0, -2)
        self.hyp_data = np.stack([endpoints, self.ideal_basis], axis=-3)

    def circle_parameter(self):
        center, radius = self.sphere_parameter()

        klein = self.space.kleinian_coords(self.endpoints)
        poincare = self.space.kleinian_to_poincare(klein)

        thetas = utils.circle_angles(center, poincare)

        return center, radius, thetas

class Hyperplane(Subspace):
    def __init__(self, space, spacelike_vector):
        self.space = space
        self.compute_ideal_basis(spacelike_vector)

    def set(self, hyp_data):
        self.hyp_data = hyp_data
        self.spacelike_vector = self.hyp_data.T[:,0,...].T
        self.ideal_basis = self.hyp_data.T[:,1:,...].T

    def compute_ideal_basis(self, spacelike_vector):
        n = self.space.dimension + 1
        transform = self.space.spacelike_to(spacelike_vector)

        if len(spacelike_vector.shape) < 2:
            spacelike_vector = np.expand_dims(spacelike_vector, axis=0)

        standard_ideal_basis = np.vstack(
            [np.ones((1, n-1)), np.eye(n - 1, n - 1, -1)]
        )
        standard_ideal_basis[n-1, n-2] = -1.

        self.ideal_basis = transform.apply(standard_ideal_basis.T).hyp_data
        self.spacelike_vector = spacelike_vector
        self.hyp_data = np.concatenate([self.spacelike_vector, self.ideal_basis],
                                       axis=-2)

    def from_reflection(space, reflection):
        try:
            matrix = reflection.matrix.swapaxes(-1, -2)
        except AttributeError:
            matrix = reflection

        #numpy's eig expects a matrix operating on the left
        evals, evecs = np.linalg.eig(matrix)

        #sometimes eigenvalues will be complex due to roundoff error
        #so we cast to reals to avoid warnings
        reflected = np.argmin(np.real(evals), axis=-1)

        #it might make sense to throw a fit if the eigenvalues reveal
        #that this isn't a reflection.

        spacelike = np.take_along_axis(
            np.real(evecs), np.expand_dims(reflected, axis=(-1,-2)), axis=-1
        )

        return Hyperplane(space, spacelike.swapaxes(-1,-2))

class TangentVector(HyperbolicObject):
    def __init__(self, space, point, vector):
        self.space = space
        self.set(point, vector)

    def set(self, hyp_data):
        self.hyp_data = hyp_data


    #this code doesn't work right now
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

class Isometry(HyperbolicObject):
    def __init__(self, space, hyp_data, column_vectors=True):
        self.space = space

        try:
            self._construct_from_object(hyp_data)
        except TypeError:
            if column_vectors:
                self.set(hyp_data.swapaxes(-1,-2))
            else:
                self.set(hyp_data)

    def set(self, hyp_data):
        self.matrix = hyp_data
        self.hyp_data = hyp_data

    def apply(self, hyp_obj):
        new_obj = copy(hyp_obj)

        try:
            #already looks like a hyperbolic object
            new_obj.set(new_obj.hyp_data @ self.matrix)
            return new_obj
        except AttributeError:
            pass

        #otherwise, it's an array of vectors which we'll interpret as
        #some kind of hyperbolic object
        return HyperbolicObject(self.space, hyp_obj @ self.matrix)

    def inv(self):
        return Isometry(self.space, np.linalg.inv(self.matrix))

    def __matmul__(self, other):
        return self.apply(other)

class HyperbolicRepresentation(representation.Representation):
    def __init__(self, space, generator_names=[]):
        self.space = space
        representation.Representation.__init__(self, generator_names)

    def __getitem__(self, word):
        matrix = self._word_value(word)
        return Isometry(self.space, matrix)

    def __setitem__(self, generator, isometry):
        try:
            super().__setitem__(generator, isometry.matrix)
            return
        except AttributeError:
            super().__setitem__(generator, isometry)

    def from_matrix_rep(space, rep):
        hyp_rep = HyperbolicRepresentation(space)
        for g, matrix in rep.generators.items():
            hyp_rep[g] = matrix

        return hyp_rep

    def isometries(self, words):
        matrix_array = np.array(
            [representation.Representation.__getitem__(self, word)
             for word in words]
        )
        return Isometry(self.space, matrix_array)

class HyperbolicSpace:
    """Class to work with the projective model for hyperbolic space,
    coordinatized as the unit circle in a standard affine chart.

    """
    def __init__(self, dimension):
        self.projective = projective.ProjectiveSpace(dimension + 1)
        self.dimension = dimension

    def minkowski(self):
        return np.diag(np.concatenate(([-1.0], np.ones(self.dimension))))

    def hyperboloid_coords(self, points):
        #last index of array as the dimension
        n = self.dimension + 1
        projective_coords, dim_index = utils.dimension_to_axis(points, n, -1)

        transposed = projective_coords.swapaxes(-1, -2)
        norms = transposed @ self.minkowski() @ projective_coords
        norms = norms.reshape(norms.shape + (1,))

        hyperbolized = projective_coords / np.sqrt(np.abs(norms))

        return hyperbolized.swapaxes(-1, dim_index)

    def kleinian_coords(self, points):
        n = self.dimension + 1
        projective_coords, dim_index = utils.dimension_to_axis(points, n, -1)

        return self.projective.affine_coordinates(points).swapaxes(-1, dim_index)

    def kleinian_to_poincare(self, points):
        euc_products = points @ points.swapaxes(-1, -2)
        euc_norms = np.diagonal(euc_products, axis1=-2, axis2=-1)

        mult_factor = 1 / (1. + np.sqrt(1 - euc_norms))

        return (points.T * mult_factor.T).T

    def poincare_to_kleinian(self, points):
        euc_products = points @ points.swapaxes(-1, -2)
        euc_norms = np.diagonal(euc_products, axis1=-2, axis2=-1)

        mult_factor = 2. / (1. + euc_norms)

        return (points.T * mult_factor.T).T

    def get_elliptic(self, block_elliptic):
        mat = scipy.linalg.block_diag([[1.0], block_elliptic])

        return Isometry(self, mat)

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
        return self.get_point(np.zeros(self.dimension))

    def get_point(self, point, coords="kleinian"):
        return Point(self, point, coords)

    def loxodromic_basis_change(self):
        basis_change = np.matrix([
            [1.0, 1.0],
            [1.0, -1.0]
        ])
        return scipy.linalg.block_diag(
            [basis_change, np.identity(self.dimension - 1)]
        )

    def get_standard_loxodromic(self, parameter):
        basis_change = self.loxodromic_basis_change()
        diagonal_loxodromic = np.diag(
            np.concatenate(([parameter, 1.0/parameter],
                       np.ones(self.dimension - 1)))
        )

        return Isometry(self,
            basis_change @ diagonal_loxodromic @ np.linalg.inv(basis_change)
        )

    def get_standard_rotation(self, angle):
        affine = scipy.linalg.block_diag(
            [utils.rotation_matrix(angle),
             np.identity(self.dimension - 2)]
        )
        return self.get_elliptic(affine)

    def regular_polygon(self, n, hyp_radius):
        #TODO: fix to return Point
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

    def timelike_to(self, v):
        lengths = utils.normsq(v, self.minkowski())
        if (lengths > 0).any():
            raise GeometryException

        return Isometry(self, utils.find_isometry(self.minkowski(), v),
                        column_vectors=False)

    def spacelike_to(self, v):
        iso = utils.find_isometry(self.minkowski(), v)

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

        return Isometry(self, p_iso, column_vectors=False)

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

    def get_standard_reflection(self):
        basis_change = self.basis_change
        diag_reflection = np.matrix([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        return basis_change @ diag_reflection @ np.linalg.inv(basis_change)
