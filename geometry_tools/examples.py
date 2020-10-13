import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Polygon

from geometry_tools.projective import ProjectivePlane

plane = ProjectivePlane()
plane.set_hyperplane_coordinates(np.array([1.0, 1.0, 1.0]))
plane.set_affine_origin([1.0, 1.0, 1.0])
plane.set_affine_direction([1.0, 0.0, 0.0], [0.0, 1.0])


pts = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])

affine_triangle_pts = plane.affine_coordinates(pts)
affine_triangle = Polygon(affine_triangle_pts, fill=False,
                          edgecolor="black")

basepoint = np.array([1.0, 1.0, 1.0]).T

scale_factor = 1.5
triangle_automorphism = np.matrix([
    [scale_factor, 0.0, 0.0             ],
    [0.0,          1.0, 0.0             ],
    [0.0,          0.0, 1 / scale_factor]
])

num_pts = 10
point_sequence = [
    (np.matmul((triangle_automorphism**k), basepoint)).T
    for k in range(-1 * num_pts, num_pts)
]
xs, ys = plane.xy_coords(point_sequence)

fig, ax = plt.subplots()

ax.add_patch(affine_triangle)
plt.plot(xs, ys, 'bo')

plt.show()
