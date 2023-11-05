import pytest
import numpy as np

from geometry_tools import utils

def test_exact_warnings():
    with pytest.raises(UserWarning):
        utils.invert(np.array([[1/2, 0.5],
                               [-1.3, 0.1]]),
                     compute_exact=True)
