import numpy as np

def inexact_type(array):
    try:
        dtype = array.dtype
    except AttributeError:
        return False

    return (not np.can_cast(dtype, int) and
            (np.can_cast(dtype, np.dtype("complex")) or
             np.can_cast(dtype, float)))

def is_linalg_type(array):
    try:
        dtype = array.dtype
    except AttributeError:
        return False

    return (np.can_cast(dtype, np.dtype("complex")) or
            np.can_cast(dtype, float))
