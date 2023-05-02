from geometry_tools import hyperbolic, coxeter, drawtools

# make the triangle group representation
triangle_group = coxeter.TriangleGroup((2,3,7))
triangle_rep = triangle_group.hyperbolic_rep()

# find a fundamental domain for the action by finding 
# fixed points of length-2 elements
vertices = triangle_rep.isometries(["ab", "bc", "ca"]).fixed_point()
fund_triangle = hyperbolic.Polygon(vertices)

# find all orientation-preserving isometries of length at most 30 using an automaton
triangle_fsa = triangle_group.automaton(even_length=True)
pos_isometries = triangle_rep.automaton_accepted(triangle_fsa, 15)

# draw the translated triangles
fig = drawtools.HyperbolicDrawing(model="poincare")
fig.draw_plane()
fig.draw_polygon(pos_isometries @ fund_triangle, 
                 facecolor="royalblue", edgecolor="none")