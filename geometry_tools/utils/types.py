import numpy as np

def inexact_type(array):
    try:
        return (not np.can_cast(array, int) and
                (np.can_cast(array, np.dtype("complex")) or
                 np.can_cast(array, float)))
    except TypeError:
        return False

def is_linalg_type(array):
    try:
        return (np.can_cast(array, np.dtype("complex")) or
                np.can_cast(array, float))
    except TypeError:
        return False
