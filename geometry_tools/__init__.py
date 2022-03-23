r"""
geometry_tools
==============

`geometry_tools` is a small Python package meant to ease the process of working with group actions on hyperbolic and projective space.

The package is mostly built on top of [numpy](invalid_url) and [matplotlib](invalid_url), and provides modules to:

- perform numerical computations with objects in hyperbolic space, in multiple models (namely the Klein, hyperboloid, projective, Poincare, and half-space models)

- work (again, numerically) with representations of finitely generated groups into O(d, 1) and GL(d, R)

- use finite-state automata to do some simple computations in word-hyperbolic groups 

- draw nice pictures in the hyperbolic plane

None of the functionality of this package is (mathematically) very deep. Mostly, the package just wraps more sophisticated tools in a way intended to make it easy to quickly draw good pictures in H^2 and H^3. Eventually I hope to extend some of the picture-drawing functionality to real projective space, but we're not there yet.

## Example usage

To draw a picture of a right-angled pentagon in the hyperbolic plane:

```python
from geometry_tools import hyperbolic, drawtools
from numpy import pi

# make a copy of the hyperbolic plane
plane = hyperbolic.HyperbolicPlane()

# make a right-pentagon
pentagon = plane.regular_polygon(5, angle=pi/2)

# draw the pentagon
figure = drawtools.HyperbolicDrawing()
figure.draw_plane()
figure.draw_polygon(pentagon, facecolor="lightblue")

figure.show()

```

This code produces:

![A right-angled pentagon in the Poincare disc model for the hyperbolic plane](right_angled_pentagon.png)

## Drawing plane tilings using `geometry_tools.automata`

To be continued...
"""
