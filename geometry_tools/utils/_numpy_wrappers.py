import numpy as np

def kernel(mat):
    kernel_dim = mat.shape[-1] - mat.shape[-2]
    _, _, v = np.linalg.svd(mat)

    return v[..., -kernel_dim:, :].swapaxes(-1, -2)
