import numpy as np

from .. import utils
if utils.SAGE_AVAILABLE:
    from ..utils import sagewrap

from ..representation import Representation

def surface_rep_I(**kwargs):
    four = utils.number(4, **kwargs)
    fourth2 = utils.power(2, 1 / four, **kwargs)

    base_ring, _ = utils.check_type(**kwargs)
    if base_ring is not None:
        fourth2 = base_ring(
            utils.power(sagewrap.SR(2), sagewrap.SR(1 / four))
        )

    pi = utils.pi(exact=(base_ring is not None))
    rot = utils.rotation_matrix(pi / 8, **kwargs)
    invrot = utils.rotation_matrix(-pi / 8, **kwargs)
    cospi8 = utils.cos(pi / 8, **kwargs)

    l = fourth2 * 2 * cospi8 + fourth2**2 + 1

    g1 = utils.array_like([
        [l, 0], [0, 1/l]
    ], **kwargs)

    g2 = invrot @ g1 @ rot
    g3 = invrot @ g2 @ rot
    g4 = invrot @ g3 @ rot

    rep = Representation(generator_names="abcd",
                         relations=["adCbADcB"],
                         **kwargs)
    rep["a"] = g1
    rep["b"] = g2
    rep["c"] = g3
    rep["d"] = g4

    return rep
