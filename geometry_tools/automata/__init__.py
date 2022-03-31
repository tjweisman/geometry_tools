r"""Work with finite-state automata.

The `automata` package provides tools meant to manipulate finite-state
automata, via the `geometry_tools.automata.fsa.FSA` class. Below, we manually
construct an automaton for the group \((\mathbb{Z}/2) * (\mathbb
{Z}/3) \simeq \mathrm{PSL}(2, \mathbb{Z})\):

```python

from geometry_tools.automata import fsa

# make the automaton from a python dictionary
z2_z3_aut = fsa.FSA({
    0: {'b': 1, 'B': 2, 'a': 3},
    1: {'a': 3},
    2: {'a': 3},
    3: {'b': 1, 'B':2}
}, start_vertices=[0])


# list accepted words 
list(z2_z3_aut.enumerate_fixed_length_paths(3))

```
	['bab', 'baB', 'Bab', 'BaB', 'aba', 'aBa']

For more details, see the documentation for `fsa`.

This package also provides a handful of word-acceptor automata for various
hyperbolic groups. You can get a list of available automata by running:

```python

from geometry_tools.automata import fsa
fsa.list_builtins()

```

The package does *not* provide any tools to produce finite-state automata from
a group presentation. You can, however, produce automata in this way by
running the [kbmag](https://gap-packages.github.io/kbmag/) program (which is
not included with `geometry_tools`). `kbmag` will produce automata files
which you can load and manipulate using `geometry_tools.automata`:

```python

from geometry_tools.automata import fsa

# "automaton_file.wa" is the output of the kbmag "autgroup" command
my_fsa = fsa.load_kbmag_file("automaton_file.wa")

# python dictionary describing the automaton
my_fsa.graph_dict

```

"""
