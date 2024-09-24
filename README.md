geometry_tools package
----------------------

`geometry_tools` is a Python package meant to help you work with and visualize
group actions on hyperbolic space and projective space.

The package is mostly built on top of [numpy](https://numpy.org/), [matplotlib]
(https://matplotlib.org/), and [scipy](https://scipy.org/). Optionally, the
package can use tools provided by [Sage](https://www.sagemath.org/) to
perform (slow) exact computations.

`geometry_tools` can help you:

- perform numerical (or sometimes, exact) computations with objects in
  hyperbolic space, in multiple models (namely the Klein, hyperboloid,
  projective, Poincare, and half-space models)

- draw nice pictures in the hyperbolic plane, the real projective plane, and the complex projective line

- work with representations of finitely generated groups into O(d, 1), GL(d, R), and GL(d, C)

- do hands-on computations with representations of Coxeter groups into GL(d, R)

- use finite-state automata to do some simple computations in word-hyperbolic groups

Some limited support for 3D figures is also provided (via matplotlib).

None of the functionality of this package is (mathematically) very deep.
Mostly, the package just wraps more sophisticated tools in a way intended to
make it easy to quickly draw good pictures in H^2, H^3, and RP^2 and CP^1.

## Installation

1. Download the package as a .zip file and extract it (or clone the repository)

2. If you have [pip](https://pip.pypa.io/en/stable/) installed, run `pip install .` from the directory where you downloaded the repository. If you don't have pip installed, you should install pip. Or, try running `python setup.py install` from the repository directory.

3. To check that it worked, run `import geometry_tools` from a python prompt.

## Documentation

You can find some documentation up at the [project site](http://www-personal.umich.edu/~tjwei/geometry_tools).
If you have questions or have found bugs, email me at [tjweisman@gmail.com](mailto:tjweisman@gmail.com).

## Credits

Code by Teddy Weisman, with contributions from Florian Stecker.