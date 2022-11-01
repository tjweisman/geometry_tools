from geometry_tools import hyperbolic, coxeter, drawtools
from geometry_tools.automata import fsa

# make the triangle group representation and load a finite-state automaton
triangle_group = coxeter.TriangleGroup((2,3,7))
triangle_rep = triangle_group.hyperbolic_rep()
triangle_fsa = triangle_group.automaton()


# find a fundamental domain for the action by finding 
# fixed points of length-2 elements
vertices = triangle_rep.isometries(["ab", "bc", "ca"]).fixed_point()
fund_triangle = hyperbolic.Polygon(vertices)

# find all orientation-preserving isometries of length at most 30
words = triangle_fsa.enumerate_words(30)
even_words = [word for word in words if len(word) % 2 == 0]
pos_isometries = triangle_rep.isometries(even_words)

# draw the translated triangles
fig = drawtools.HyperbolicDrawing(model="poincare")
fig.draw_plane()
fig.draw_polygon(pos_isometries @ fund_triangle, 
                 facecolor="royalblue", edgecolor="none")

fig.show()