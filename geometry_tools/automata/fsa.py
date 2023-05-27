"""Work with finite-state automata.

Finite-state automata are really just finite directed labeled graphs,
subject to the constraint that each vertex `v` has at most one
outgoing edge with a given label. One or more of the vertices are
"start states;" a word `w` in the set of edge labels is "accepted" if
there is an edge path in the graph (starting at a start state) whose
sequence of edge labels gives `w`.

This module provides the `FSA` class, which can be used to perform
simple tasks with finite-state automata. A common use-case is
enumerating the list of words accepted by a particular automaton:

```python
from geometry_tools.automata import fsa

# load a word-acceptor automaton for the (3,3,4) triangle group
cox_aut = fsa.load_builtin("cox334.wa")

# list all the words accepted by this automaton with length at most 3.
list(cox_aut.enumerate_words(3))

```
     ['',
     'a',
     'b',
     'c',
     'ab',
     'ac',
     'ba',
     'bc',
     'ca',
     'cb',
     'aba',
     'abc',
     'aca',
     'acb',
     'bac',
     'bca',
     'bcb',
     'cab',
     'cac',
     'cba']

    """

import pkg_resources
import copy
from collections import deque, defaultdict

from . import kbmag_utils, gap_parse
from .. import utils

BUILTIN_DIR = "builtin"


class FSAException(Exception):
    pass

class FSA:
    """FSA: a finite state-automaton.

    The underlying data structure of a finite-state automaton is a
    python dictionary of the form:

    ```
    {
      vertex1: {label_a: neighbor_a, label_b: neighbor_b, ...},
      vertex2: ....,
    }
    ```

    That is, `vertex1` is connected to `neighbor_a` by an edge with
    label `label_a`, etc.

    The automaton can be constructed by passing it a dictionary of
    this form, or by adding vertices/edges individually through the
    built-in methods.

    """
    def __init__(self, vert_dict={}, start_vertices=[], graph_dict=True):
        """

        Parameters
        ----------
        vert_dict : dict
            dictionary of vertices used to build the automaton. By
            default, this dictionary should have the format specified
            above.

        start_vertices : list
            list of vertices which are valid start states
            for this automaton. Currently, the `FSA` class will not
            correctly handle lists of length > 1.

        graph_dict : bool
            If `True` (the default), expect a dictionary of the format
            listed above. If `False`, instead expect a dictionary with format:

            ```
            {
              vertex1: {neighbor_a: [neighbor_a_label1, neighbor_a_label2, ...],
                        neighbor_b: [neighbor_b_label1, neighbor_b_label2, ...], ....},
              vertex2: ....,
            }
            ```
        """
        if graph_dict:
            self._from_graph_dict(vert_dict)
        else:
            self._out_dict = copy.deepcopy(vert_dict)
            self._build_in_dict()
            self._build_graph_dict()

        self.start_vertices = start_vertices

    def _from_graph_dict(self, graph_dict):
        self._graph_dict = copy.deepcopy(graph_dict)
        hidden = _hidden_vertices(graph_dict)
        for v in hidden:
            self._graph_dict[v] = {}

        out_dict = {v : {} for v in self._graph_dict}

        for v, neighbors in self._graph_dict.items():
            out_dict[v] = defaultdict(list)
            for label, neighbor in neighbors.items():
                out_dict[v][neighbor].append(label)

        self._out_dict = out_dict
        self._build_in_dict()

    def _build_graph_dict(self):
        label_dict = {v:{} for v in self._out_dict}
        for v, neighbors_out in self._out_dict.items():
            for w, labels in neighbors_out.items():
                for label in labels:
                    label_dict[v][label] = w
        self._graph_dict = label_dict

    def _build_in_dict(self):
        in_dict = defaultdict(dict)
        for v, neighbors_out in self._out_dict.items():
            for w, labels in neighbors_out.items():
                in_dict[w][v] = labels
        self._in_dict = in_dict

    def __str__(self):
        return "FSA with dict:\n{}".format(self._graph_dict.__str__())

    def __repr__(self):
        return "FSA({})".format(self._graph_dict.__repr__())

    @property
    def out_dict(self):
        return self._out_dict

    @property
    def graph_dict(self):
        return self._graph_dict

    @property
    def in_dict(self):
        return self._in_dict

    def edges_out(self, vertex):
        """Get the list of edges directed away from a vertex.
        """
        for w, labels in self._out_dict[vertex].items():
            for label in labels:
                yield (vertex, w, label)

    def edges_in(self, vertex):
        """Get the list of edges directed towards a vertex."""
        for w, labels in self._in_dict[vertex].items():
            for label in labels:
                yield (w, vertex, label)

    def neighbors_out(self, vertex):
        """Get the list of outward neighbors of a vertex."""
        return self._out_dict[vertex].keys()

    def neighbors_in(self, vertex):
        """Get the list of inward neighbors of a vertex."""
        return self._in_dict[vertex].keys()

    def edge_label(self, tail, head):
        """Get the label of an edge from its tail and head vertex.

        This function will raise an exception if there is not a unique
        edge between `tail` and `head`.

        """
        if len(self._out_dict[tail][head]) == 1:
            return self._out_dict[tail][head][0]
        else:
            raise ValueError("ambiguous edge removal: there is not exactly"
                            f" one edge between {tail} and {head}" )

    def edge_labels(self, tail, head):
        """Get the list of labels associated to all the edges between `tail`
        and `head`.

        """
        return self._out_dict[tail][head]

    def add_vertices(self, vertices):
        """Add vertices to the FSA.
        """
        for v in vertices:
            if v not in self._out_dict:
                self._out_dict[v] = {}
                self._in_dict[v] = {}
                self._graph_dict[v] = {}

    def add_edges(self, edges, elist=False,
                  ignore_redundant=True):
        """Add edges to the FSA.

        Parameters
        ----------
        edges : list
            a list of tuples of the form `(tail, head, label)`,
            specifying an edge to add.

        elist : bool
            if `True`, then `label` should be interpreted as a single
            edge label. Otherwise, `label` is interpreted as a list of
            labels for multiple edges to be added between vertices.

        """
        for e in edges:
            tail, head, label = e

            if head not in self._out_dict[tail]:
                self._out_dict[tail][head] = []
                self._in_dict[head][tail] = []

            if ignore_redundant and label in self._out_dict[tail][head]:
                continue

            if elist:
                self._out_dict[tail][head] += label
                self._in_dict[head][tail] += label
                for l in labels:
                    self._graph_dict[tail][l] = head

            else:
                self._out_dict[tail][head].append(label)
                self._in_dict[head][tail].append(label)
                self._graph_dict[tail][label] = head

    def delete_vertices(self, vertices):
        """Delete several vertices from the FSA.
        """
        for v in vertices:
            self.delete_vertex(v)

    def delete_vertex(self, vertex):
        """Delete a single vertex from the FSA.
        """
        for w in self.neighbors_out(vertex):
            self._in_dict[w].pop(vertex)
        for w in self.neighbors_in(vertex):
            self._out_dict[w].pop(vertex)

            for lab in list(self._graph_dict[w]):
                if self._graph_dict[lab] == vertex:
                    self._graph_dict.pop(lab)

        self._out_dict.pop(vertex)
        self._in_dict.pop(vertex)
        self._graph_dict.pop(vertex)

    def vertices(self):
        return self._out_dict.keys()

    def edges(self):
        """Get all of the edges of this FSA

        Yields
        ------
        tuple
            An ordered pair of the form `(tail, head)` for each edge
            in the automaton

        """
        for v, nbrs in self._out_dict.items():
            for w in nbrs:
                yield (v, w)

    def recurrent(self, inplace=False):
        """Find a version of the automaton which is recurrent, i.e. which has
        no dead ends.

        Parameters
        ----------
        inplace : bool
            if `True`, modify the automaton in-place. Otherwise,
            return a recurrent copy.

        Returns
        -------
        FSA or None
            A recurrent version of the automaton if `inplace` is
            `True`, otherwise `None`.

        """

        to_modify = self
        if inplace:
            to_modify = copy.deepcopy(self)

        still_pruning = True
        while still_pruning:
            still_pruning = False
            for v in to_modify._out_dict.keys():
                if (len(to_modify._out_dict[v]) == 0 or
                    len(to_modify._in_dict[v]) == 0):
                    to_modify.delete_vertex(v)
                    still_pruning = True

        if not inplace:
            return to_modify

    def remove_long_paths(self, root=None, edge_ties=True,
                          return_distances=False):
        """Find a version of the automaton which does not backtrack.

        This function is the opposite of `make_recurrent`: it finds a
        subgraph of the automaton which is the union of
        shortest-length paths from the root vertex to each node. This
        will be a directed acyclic graph (in fact, a tree if
        `edge_ties` is false).

        It is mostly useful for visualizing the structure of the
        automaton.

        Parameters
        ----------
        edge_ties : bool
            if `True`, if two edges give rise to paths of the
            same length, include both of them in the new graph.

    """
        H = FSA({})
        H.add_vertices(self.vertices())

        #Dijkstra's algorithm!
        distance = {}
        marked = {v:False for v in self.vertices()}

        if root is None:
            root = self.start_vertices[0]

        marked[root] = True
        distance[root] = 0
        vertex_queue = deque([root])
        while len(vertex_queue) > 0:
            v = vertex_queue.popleft()

            to_visit = [w for w in self.neighbors_out(v) if not marked[w]]
            for w in to_visit:
                marked[w] = True
                distance[w] = distance[v] + 1

            short_nbrs = to_visit
            if edge_ties:
                short_nbrs = [w for w in self.neighbors_out(v)
                               if distance[w] == distance[v] + 1]

            H.add_edges([(v, w, self.edge_labels(v,w)) for w in short_nbrs],
                        elist= True)
            vertex_queue.extend(to_visit)

        return H

    def follow_word(self, word, start_vertex=None):
        """Find the final state of the automaton when it tries to accept a word

        Parameters
        ----------
        word : string
            The word the automaton should try to accept

        start_vertex: object
            The start state for the automaton. If `None` (the
            default), use the automaton's default start state.

        Returns
        -------
        object
            The state of the automaton when it accepts `word`

        Raises
        ------
        FSAException
            Raised if word is not accepted by this automaton.

        """

        if start_vertex is None:
            start_vertex = self.start_vertices[0]
        vertex = start_vertex

        for letter in word:
            try:
                vertex = self.graph_dict[vertex][letter]
            except KeyError:
                raise FSAException(
                    f"The word '{word}' is not accepted by the automaton."
                )

        return vertex

    def enumerate_fixed_length_paths(self, length, start_vertex=None,
                                     with_states=False):
        """Enumerate all words of a fixed length accepted by the automaton,
        starting at a given state.

        Parameters
        ----------
        length : int
            the length of the words we want to enumerate

        start_vertex : object
            which vertex to start at. If `None`, use the
            default starting state for the automaton.

        with_states : bool
            if `True`, also yield the state of the
            automaton at each accepted word.

        Yields
        ------
        string or (string, object)
            If `with_states` is false, yield the sequence of words
            accepted (in some arbitrary order). Otherwise, yield pairs
            of the form `(word, end_state)`, where `end_state` is the
            final state the automaton reaches when accepting `word`.
        """
        if start_vertex is None:
            start_vertex = self.start_vertices[0]

        if length <= 0:
            if with_states:
                yield ("", start_vertex)
            else:
                yield ""
        else:
            for word, vertex in self.enumerate_fixed_length_paths(
                    length - 1, start_vertex=start_vertex, with_states=True):
                for label, neighbor in self._graph_dict[vertex].items():
                    if with_states:
                        yield (word + label, neighbor)
                    else:
                        yield word + label

    def enumerate_words(self, max_length, start_vertex=None,
                        with_states=False):
        """Enumerate all words up to a given length accepted by the automaton.

        Parameters
        ----------
        max_length : int
            maximum length of a word to enumerate
        start_vertex : object
            the start vertex for accepting words. If `None`, use the
            automaton's start vertex
        with_states : bool
            if `True`, also yield the state of the automaton at each
            accepted word

        Yields
        ------
        string or (string, object)
            If `with_states` is false, yield the sequence of words
            accepted (in some arbitrary order). Otherwise, yield pairs
            of the form `(word, end_state)`, where `end_state` is the
            final state the automaton reaches when accepting `word`.
        """

        for i in range(max_length + 1):
            for word in self.enumerate_fixed_length_paths(
                    i, start_vertex=start_vertex, with_states=with_states):
                yield word

    def rename_generators(self, rename_map, inplace=True):
        """Get a new automaton whose edge labels have been renamed by a
        specified map.

        Parameters
        ----------
        rename_map : dict
            dictionary giving the mapping from old labels to new
            labels

        inplace : bool
            If True (the default), modify the current automaton
            in-place. If False, construct and return a new automaton
            and leave this one unchanged.

        Returns
        -------
        FSA or None
            If inplace is False, return a new automaton with renamed
            edge labels. Otherwise, None.

        """
        new_dict = {
            vertex: {
                rename_map[label]: neighbor
                for label, neighbor in neighbors.items()
            }
            for vertex, neighbors in self._graph_dict.items()
        }
        if inplace:
            self._from_graph_dict(new_dict)
        else:
            return FSA(new_dict, self.start_vertices)

    def automaton_multiple(self, multiple):
        """Get a new automaton, which accepts words in the language of this
        automaton whose length is a multiple of k for some fixed k.

        Parameters
        ----------
        multiple : int
            Integer k such that every word accepted by the new
            automaton has length kn for some n.

        Returns
        -------
        FSA
            A finite-state automaton which accepts exactly the words
            accepted by self whose length is a multiple of k.

        """
        to_visit = deque(self.start_vertices)
        visited = {
            v : False for v in self.vertices()
        }
        new_automaton = FSA(start_vertices=self.start_vertices)
        while(len(to_visit) > 0):
            v = to_visit.popleft()
            visited[v] = True
            new_automaton.add_vertices([v])

            for word, neighbor in self.enumerate_fixed_length_paths(
                multiple,
                start_vertex=v,
                with_states=True
            ):
                new_automaton.add_vertices([neighbor])
                new_automaton.add_edges([(v, neighbor, word)],
                                        elist=False)

                if not visited[neighbor]:
                    to_visit.append(neighbor)

        return new_automaton

    def even_automaton(self):
        """Get a new automaton which accepts exactly the the words accepted by
        this automaton which have even length.

        Alias for automaton_multiple(2).

        Returns
        -------
        FSA
            A finite-state automaton which accepts exactly the words
            accepted by self whose length is even.

        """
        return self.automaton_multiple(2)

    def accepts(self, word, start_vertex=None):
        """Determine if this automaton accepts a given word.

        Parameters
        ----------
        word : string
            word to test for acceptance
        start_vertex : object
            The start state for the automaton. If `None` (the
            default), use the automaton's default start state.

        Returns
        -------
        bool
            True if word is accepted by the automaton, False
            otherwise.

        """
        try:
            self.follow_word(word, start_vertex)
        except FSAException:
            return False

        return True


def load_kbmag_file(filename) -> FSA:

    """Build a finite-state automaton from a GAP record file produced by
    the kbmag program.

    Parameters
    ----------
    filename : string
        the GAP record file containing automaton data.

    Returns
    -------
    FSA
        The automaton described by the
        file.

    """
    gap_record = gap_parse.load_record_file(filename)
    return _from_gap_record(gap_record)


def _from_gap_record(gap_record) -> FSA:
    """Load an automaton from a GAP record

    Parameters
    ----------
    gap_record : dict
        Preparsed GAP record to load

    Returns
    -------
    FSA
        finite-state automaton represented by this record

    """

    for fsa_dict in gap_record.values():
        if fsa_dict["isFSA"] == "true":
            labels = fsa_dict["alphabet"]["names"]
            transitions = fsa_dict["table"]["transitions"]
            initial = fsa_dict["initial"]

            out_dict = kbmag_utils.build_dict(
                transitions, labels, to_filter=[0]
            )

            return FSA(out_dict, start_vertices=initial)


def load_builtin(filename):
    """Load a finite-state automaton built in to the automata subpackage.

    Parameters
    ----------
    filename : string
        Name of the automaton file to load

    Returns
    -------
    FSA
        finite-state automaton read from this file

    """
    aut_string = pkg_resources.resource_string(
        __name__, "{}/{}".format(BUILTIN_DIR, filename)
    )
    record, _ = gap_parse.parse_record(aut_string.decode('utf-8'))
    return _from_gap_record(record)


def list_builtins():
    """Return a list of all the automata included with the automata
    subpackage.

    """
    return pkg_resources.resource_listdir(__name__, BUILTIN_DIR)

def free_automaton(generating_set):
    """Return an automaton accepting freely reduced words in a set of
    generators

    Parameters
    ----------
    generating_set : iterable of strings
        Names of the generators for this free group

    Returns
    -------
    FSA
        Automaton accepting freely reduced words in the generators
        (and their inverses)

    """
    generators = list(generating_set) + [
        utils.invert_gen(g) for g in generating_set
    ]
    graph = {
        g : {h:h for h in generators
             if utils.invert_gen(h) != g}
        for g in [''] + generators
    }
    fsa = FSA(graph, start_vertices=[''])
    return fsa

def _hidden_vertices(graph_dict):
    vertices = list(graph_dict)
    hidden = []
    for v, neighbors in graph_dict.items():
        for _, neighbor in neighbors.items():
            if (neighbor not in vertices and
                neighbor not in hidden):
                hidden.append(neighbor)

    return hidden
