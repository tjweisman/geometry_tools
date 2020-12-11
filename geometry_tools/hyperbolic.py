from copy import copy

import numpy as np
import scipy
from scipy.optimize import fsolve

from geometry_tools import projective, representation, utils

#TODO: raise exceptions when nonsensical geometry data is passed to
#these classes

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

    def __getitem__(self, item):
        return self.__class__(self.space, self.hyp_data[item])

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
    def __init__(self, space, point, coords="projective"):
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

    def distance(self, other):
        """compute all pairwise distances (w.r.t. last two dimensions) between
        self and other"""
        products = (
            self.hyp_data @ self.space.minkowski() @ other.hyp_data.swapaxes(-1, -2)
        )
        return np.arccosh(np.abs(products))

    def origin_to(self):
        """return an isometry taking the origin to this point"""
        return self.space.timelike_to(self.hyp_data)

    def unit_tangent_towards(self, other):
        diff = other.hyp_data - self.hyp_data
        return TangentVector(self.space, self, diff).normalized()


class DualPoint(HyperbolicObject):
    pass

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
        """construct a hyperplane which is the fixpoint set of a given
        reflection"""
        try:
            matrix = reflection.matrix.swapaxes(-1, -2)
        except AttributeError:
            matrix = reflection

        #numpy's eig expects a matrix operating on the left
        evals, evecs = np.linalg.eig(matrix)

        #sometimes eigenvalues will be complex due to roundoff error
        #so we cast to reals to avoid warnings. TODO: throw an error
        #message if imaginary part is large.
        reflected = np.argmin(np.real(evals), axis=-1)

        spacelike = np.take_along_axis(
            np.real(evecs), np.expand_dims(reflected, axis=(-1,-2)), axis=-1
        )

        return Hyperplane(space, spacelike.swapaxes(-1,-2))

class TangentVector(HyperbolicObject):
    def __init__(self, space, point_data, vector=None):
        self.space = space
        if vector is None:
            try:
                self._construct_from_object(point_data)
                return
            except TypeError:
                pass

            self.set(point_data)
            return

        self.compute_data(point_data, vector)

    def compute_data(self, point, vector):
        #wrapping these as hyperbolic objects first
        pt = Point(self.space, point)
        vec = DualPoint(self.space, vector)

        projected = self.space.project_to_hyperboloid(pt.hyp_data, vec.hyp_data)

        self.set(np.stack([pt.hyp_data, projected], axis=-2))

    def set(self, hyp_data):
        self.hyp_data = hyp_data
        self.point = self.hyp_data[..., 0, :]
        self.vector = self.hyp_data[..., 1, :]

    def normalized(self):
        normed_vec = utils.normalize(self.vector, self.space.minkowski())
        return TangentVector(self.space, self.point, normed_vec)

    def origin_to(self, force_oriented=False):
        normed = utils.normalize(self.hyp_data, self.space.minkowski())
        isom = utils.find_isometry(self.space.minkowski(), normed,
                                   force_oriented)

        return Isometry(self.space, isom, column_vectors=False)

    def angle(self, other):
        v1 = self.space.project_to_hyperboloid(self.point, self.normalized().vector)
        v2 = self.space.project_to_hyperboloid(self.point, other.normalized().vector)

        product = utils.apply_bilinear(v1, v2, self.space.minkowski())
        return np.arccos(product)

    def point_along(self, distance):
        kleinian_shape = list(self.point.shape)
        kleinian_shape[-1] -= 1

        kleinian_pt = np.zeros(kleinian_shape)
        kleinian_pt[..., 0] = self.space.hyp_to_affine_dist(distance)

        basepoint = Point(self.space, kleinian_pt, coords="kleinian")

        return self.origin_to().apply(basepoint, "elementwise")

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

    def _apply_to_data(self, hyp_data, broadcast):
        if broadcast == "matmul":
            product = hyp_data @ self.matrix
        elif broadcast == "pairwise":
            product = utils.pairwise_matrix_product(hyp_data, self.matrix)
        elif broadcast == "elementwise":
            exp_data = np.expand_dims(hyp_data, axis=-2)
            product = np.squeeze(exp_data @ self.matrix, axis=-2)

        return product

    def apply(self, hyp_obj, broadcast="matmul"):
        new_obj = copy(hyp_obj)

        try:
            hyp_data = new_obj.hyp_data
            product = self._apply_to_data(new_obj.hyp_data, broadcast)
            new_obj.set(product)
            return new_obj
        except AttributeError:
            pass

        #otherwise, it's an array of vectors which we'll interpret as
        #some kind of hyperbolic object
        product = self._apply_to_data(hyp_obj, broadcast)
        return HyperbolicObject(self.space, product)

    def inv(self):
        return Isometry(self.space, np.linalg.inv(self.matrix),
                        column_vectors=False)

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
            super().__setitem__(generator, isometry.matrix.T)
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
    def __init__(self, dimension):
        self.projective = projective.ProjectiveSpace(dimension + 1)
        self.dimension = dimension

    def minkowski(self):
        return np.diag(np.concatenate(([-1.0], np.ones(self.dimension))))

    def hyperboloid_coords(self, points):
        #last index of array as the dimension
        n = self.dimension + 1
        projective_coords, dim_index = utils.dimension_to_axis(points, n, -1)

        hyperbolized = utils.normalize(projective_coords, self.minkowski())

        return hyperbolized.swapaxes(-1, dim_index)

    def kleinian_coords(self, points):
        n = self.dimension + 1
        projective_coords, dim_index = utils.dimension_to_axis(points, n, -1)

        return self.projective.affine_coordinates(points).swapaxes(-1, dim_index)

    def kleinian_to_poincare(self, points):
        euc_norms = utils.normsq(points)
        mult_factor = 1 / (1. + np.sqrt(1 - euc_norms))

        return (points.T * mult_factor.T).T

    def poincare_to_kleinian(self, points):
        euc_norms = utils.normsq(points)
        mult_factor = 2. / (1. + euc_norms)

        return (points.T * mult_factor.T).T

    def get_elliptic(self, block_elliptic):
        mat = scipy.linalg.block_diag(1.0, block_elliptic)

        return Isometry(self, mat)

    def project_to_hyperboloid(self, basepoint, tangent_vector):
        return tangent_vector - utils.projection(
            tangent_vector, basepoint, self.minkowski())

    def get_origin(self, shape=()):
        return self.get_point(np.zeros(shape + (self.dimension,)))

    def get_base_tangent(self, shape=()):
        origin = self.get_origin(shape)
        vector = np.zeros(() + (self.dimension + 1,))
        vector[..., 1] = 1

        return TangentVector(self, origin, vector)

    def get_point(self, point, coords="kleinian"):
        return Point(self, point, coords)

    def hyp_to_affine_dist(self, r):
        return (np.exp(2 * r) - 1) / (1 + np.exp(2 * r))

    def loxodromic_basis_change(self):
        basis_change = np.matrix([
            [1.0, 1.0],
            [1.0, -1.0]
        ])
        return scipy.linalg.block_diag(
            basis_change, np.identity(self.dimension - 1)
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
            utils.rotation_matrix(angle),
            np.identity(self.dimension - 2)
        )
        return self.get_elliptic(affine)

    def regular_polygon(self, n, hyp_radius):
        radius = np.array(hyp_radius)
        tangent = self.get_base_tangent(radius.shape).normalized()
        start_vertex = tangent.point_along(radius)

        start_vertex.set(np.expand_dims(start_vertex.hyp_data, axis=-2))

        cyclic_rep = HyperbolicRepresentation(self)
        cyclic_rep["a"] = self.get_standard_rotation(2 * np.pi / n)

        words = ["a" * i for i in range(n)]
        mats = cyclic_rep.isometries(words)

        vertices = mats.apply(start_vertex, "pairwise")
        vertices.set(np.squeeze(vertices.hyp_data, axis=-2))

        return vertices

    def polygon_interior_angle(self, n, hyp_radius):
        polygon = self.regular_polygon(n, hyp_radius)

        tv1 = polygon[..., 0, :].unit_tangent_towards(polygon[..., -1, :])
        tv2 = polygon[..., 0, :].unit_tangent_towards(polygon[..., 1, :])

        return tv1.angle(tv2)

    def timelike_to(self, v, force_oriented=False):
        lengths = utils.normsq(v, self.minkowski())
        if (lengths > 0).any():
            raise GeometryException

        return Isometry(self, utils.find_isometry(self.minkowski(),
                                                  v, force_oriented),
                        column_vectors=False)

    def spacelike_to(self, v, force_oriented=False):
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

        if force_oriented:
            p_iso = utils.make_orientation_preserving(p_iso)

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
        return Isometry(self,
                        basis_change @ diag_reflection @ np.linalg.inv(basis_change)
        )
