import numpy as np

def svd_kernel(mat, assume_full_rank=False, matching_rank=True,
               tolerance=1e-8, with_dimensions=False,
               with_loc=False):

    if assume_full_rank and not matching_rank:
        raise ValueError("matching_rank must be True if assume_full_rank is True")

    _, s, v = np.linalg.svd(mat)

    min_kernel_dim = mat.shape[-1] - mat.shape[-2]
    if assume_full_rank:
        kernel_dim = min_kernel_dim
    else:
        small_sv = s < tolerance
        kernel_dims = min_kernel_dim + np.count_nonzero(small_sv, axis=-1)
        actual_ranks_match = np.all(
            kernel_dims == np.atleast_1d(kernel_dims)[0]
        )

        if matching_rank and not actual_ranks_match:
            raise ValueError(
                "Input matrices do not have matching rank. Try calling "
                "this function with matching_rank=False."
            )

        kernel_dim = np.atleast_1d(kernel_dims)[0]

    if matching_rank:
        return v[..., -kernel_dim:, :].swapaxes(-1, -2)

    possible_dims = np.unique(kernel_dims, )
    kernel_bases = []
    kernel_dim_loc = []

    for kernel_dim in possible_dims:
        where_dim = (kernel_dims == kernel_dim)
        kernel_bases.append(
            v[where_dim, -kernel_dim:, :].swapaxes(-1, -2)
        )
        kernel_dim_loc.append(where_dim)

    # flat is better than nested
    if not with_loc and not with_dimensions:
        return tuple(kernel_bases)

    if with_dimensions and not with_loc:
        return (possible_dims, tuple(kernel_bases))

    if with_loc and not with_dimensions:
        return (tuple(kernel_bases), tuple(kernel_dim_loc))

    return (possible_dims, tuple(kernel_bases), tuple(kernel_dim_loc))
