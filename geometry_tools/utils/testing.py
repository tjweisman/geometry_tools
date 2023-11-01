import numpy as np

def same_none(n1, n2):
    return (n1 is None and n2 is None) or (n1 is not None and n2 is not None)

def assert_numpy_equivalent(obj1, obj2):
    assert obj1.proj_data.dtype == obj2.proj_data.dtype
    assert obj1.shape == obj2.shape

    assert obj1.unit_ndims == obj2.unit_ndims
    assert obj1.aux_ndims == obj2.aux_ndims
    assert obj1.dual_ndims == obj2.dual_ndims

    assert same_none(obj1.aux_data, obj2.aux_data)
    assert same_none(obj1.dual_data, obj2.dual_data)

    assert np.allclose(
        obj1.proj_data,
        obj2.proj_data
    )

    if obj1.aux_data is not None:
        assert np.allclose(
            obj1.aux_data,
            obj2.aux_data
        )

    if obj1.dual_data is not None:
        assert np.allclose(
            obj1.dual_data,
            obj2.dual_data
        )
