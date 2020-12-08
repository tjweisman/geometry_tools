from collections import defaultdict

import numpy as np
from scipy.spatial import ConvexHull

def make_cycle(simplices):
    if len(simplices) == 0:
        return
    neighbors = defaultdict(list)
    for simplex in simplices:
        neighbors[simplex[0]].append(simplex[1])
        neighbors[simplex[1]].append(simplex[0])

    indices = []
    prev_point = None
    current_point = simplices[0][0]
    first_point = current_point
    cycle_complete = False
    while not cycle_complete:
        indices.append(current_point)
        n1, n2 = tuple(neighbors[current_point])
        if n1 != prev_point:
            prev_point = current_point
            current_point = n1
        else:
            prev_point = current_point
            current_point = n2

        if current_point == first_point:
            cycle_complete = True
    return indices

def convex_polygon(points):
    hull = ConvexHull(points)
    return np.array([points[index] for index in make_cycle(hull.simplices)])
