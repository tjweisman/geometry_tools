import numpy as np

def inexact_type(array):
    #TODO: check some common sage type conversions to make sure this doesn't bug out
    nparr = np.array(array)
    try:
        return (not np.can_cast(nparr, int) and
                (np.can_cast(nparr, np.dtype("complex")) or
                 np.can_cast(nparr, float)))
    except TypeError:
        return False

def is_linalg_type(array):
    #TODO: check some common sage type conversions to make sure this doesn't bug out
    nparr = np.array(array)
    try:
        return (np.can_cast(nparr, np.dtype("complex")) or
                np.can_cast(nparr, float))
    except TypeError:
        return False
