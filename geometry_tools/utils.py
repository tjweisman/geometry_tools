import numpy as np
import scipy

def permutation_matrix(permutation):
    n = len(permutation)
    p_mat = np.zeros((n, n))
    for i,j in enumerate(permutation):
        p_mat[i,j] = 1.

    return p_mat

def swap_matrix(i, j, n):
    permutation = list(range(n))
    permutation[i] = j
    permutation[j] = i
    return permutation_matrix(permutation)

def indefinite_orthogonalize(form, matrix):
    #this is really inefficient for large arrays, and probably unstable.
    n, m = matrix.shape
    if m < 1:
        return
    orthogonalized = np.array(matrix[:,0]).reshape(n,1)
    for i in range(1, m):
        complement = scipy.linalg.null_space(orthogonalized.T @ form)
        orthogonalized = np.column_stack([orthogonalized, complement[:,0]])

    norm_sq = 1./np.diagonal(orthogonalized.T @ form @ orthogonalized)

    return orthogonalized @ np.diag(np.sqrt(np.abs(norm_sq)))
