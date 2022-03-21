"""fsa.py: tools to work with finite-state automata

This module provides the FSA class, which can be used to facilitate
the construction of finite-state automata and perform tasks like
enumerating accepted words.

Finite-state automata are really just finite directed labeled graphs,
subject to the constraint that each vertex v has at most one outgoing
edge with a given label. One or more of the vertices are "start
states;" a word w in the set of edge labels is "accepted" if there is
an edge path in the graph (starting at a start state) whose sequence
of edge labels gives w.

"""

import pkg_resources
import copy
from collections import deque, defaultdict

from . import kbmag_utils, gap_parse

BUILTIN_DIR = "builtin"

class FSAException(Exception):
    pass

class FSA:
    """FSA: a finite state-automaton.

    The underlying data structure of a finite-state automaton is a
    python dictionary of the form:

    ```
    { vertex1: {label_a: neighbor_a, label_b: neighbor_b, ...},
      vertex2: ....,
    }
    ```

    That is, vertex1 is connected to neighbor_a by an edge with label
    label_a, etc.

    The automaton can be constructed by passing it a dictionary of
    this form, or by adding vertices/edges individually through the
    built-in methods.

    """

    def __init__(self, vert_dict={}, start_vertices=[], graph_dict=True):
        """
        Parameters
        ----------
        vert_dict: dictionary of vertices used to build the
        automaton. By default, this dictionary should have format:

        ```
        { vertex1: {label_a: neighbor_a, label_b: neighbor_b, ...},
          vertex2: ....,
        }
        ```

        start_vertices: list of vertices which are valid start states
        for this automaton. Currently, the FSA class will not
        correctly handle lists of length > 1.

        graph_dict: If True (the default), expect a dictionary of the format
        listed above. If False, instead expect a dictionary with format:

        ```
        { vertex1: {neighbor_a: [neighbor_a_label1, neighbor_a_label2, ...],
                    neighbor_b: [neighbor_b_label1, neighbor_b_label2, ...],
                    ....},
          vertex2: ....,
        }
        ```

        """
        if graph_dict:
            self._from_graph_dict(vert_dict)
        else:
            self._out_dict = copy.deepcopy(vert_dict)
            self._build_in_dict()
            self._build_graph_dict

        self.start_vertices = start_vertices

    def _from_graph_dict(self, graph_dict):
        self._graph_dict = copy.deepcopy(graph_dict)
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
        in_dict = {v:{} for v in self._out_dict}
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
        edge between this pair of vertices.

        """
        if len(self._out_dict[tail][head]) == 1:
            return self._out_dict[tail][head][0]
        else:
            raise FSAException("ambiguous: there is not exactly"
                            " one edge between these vertices!" )

    def edge_labels(self, tail, head):
        """Get the list of labels associated to all the edges between a pair
        of vertices.
        """
        return self._out_dict[tail][head]

    def add_vertices(self, vertices):
        """Add an isolated vertex to the FSA.
        """
        for v in vertices:
            if v not in self._out_dict:
                self._out_dict[v] = {}
                self._in_dict[v] = {}
                self._graph_dict[v] = {}

    def add_edges(self, edges, elist=False):
        """Add edges to the FSA.


        edges: a list of tuples of the form `(tail, head, label)`,
        specifying an edge to add.

        elist: if True, then label should be interpreted as a single
        edge label. Otherwise, label is interpreted as a list of
        labels for multiple edges to be added between vertices.

        """
        for e in edges:
            tail, head, label = e

            if head not in self._out_dict[tail]:
                self._out_dict[tail][head] = []
                self._in_dict[head][tail] = []

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
        """Delete a vertex from the FSA.
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
        for v, nbrs in self._out_dict.items():
            for w in nbrs:
                yield (v,w)

    def recurrent(self, inplace=False):
        """Find a version of the automaton which is recurrent, i.e. which has
        no dead ends.

        inplace: if True, modify the automaton in-place. Otherwise,
        return a recurrent copy.

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

        This function is the opposite of make_recurrent: it finds a
        subgraph of the automaton which is the union of
        shortest-length paths from the root vertex to each node. This
        will be a directed acyclic graph (in fact, a tree if
        edge_ties is false).

        It is mostly useful for visualizing the structure of the
        automaton.

        edge_ties: if True, if two edges give rise to paths of the
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
        """Find the final state of the automaton when it tries to accept a
        given word.

        If start_vertex is None, use the default starting state for
        the automaton.

        """
        if start_vertex is None:
            start_vertex = start_vertices[0]
        vertex = start_vertex

        for letter in word:
            vertex = graph_dict[vertex][letter]

        return vertex

    def enumerate_fixed_length_paths(self, length, start_vertex=None,
                                     with_states=False):
        """Enumerate all words of a fixed length accepted by the automaton,
        starting at a given state.

        Required arguments:
        ------------------
        length: the length of the words we want to enumerate

        Keyword arguments:
        -------------------
        start_vertex: which vertex to start at. If `None`, use the
        default starting state for the automaton.

        with_states: if True, also yield the state of the
        automaton at each accepted word.

        Return:
        -------
        If `with_states` is false, return a generator yielding the
        sequence of words accepted (in some arbitrary
        order). Otherwise, return a generator yielding pairs of the
        form `(word, end_state)`, where `end_state` is the final state
        the automaton reaches when accepting `word`.

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

    def enumerate_words(self, max_length, start_vertex=None, with_states=False):
        """Enumerate all words up to a given length accepted by the
        automaton.

        Required arguments:
        -------------------
        max_length: maximum length of a word to enumerate

        Keyword arguments:
        ------------------
        start_vertex: the start vertex for accepting words. If None,
        use the automaton's starting vertex.

        with_states: if `True`, also yield the state of the
        automaton at each accepted word.

        Return:
        -------
        If with_states is false, return a generator yielding the
        sequence of words accepted (in some arbitrary
        order). Otherwise, return a generator yielding pairs of the
        form (word, end_state), where end_state is the final state the
        automaton reaches when accepting word.

        """

        for i in range(max_length + 1):
            for word in self.enumerate_fixed_length_paths(
                    i, start_vertex=start_vertex, with_states=with_states):
                yield word

def from_kbmag_file(filename):
    """Build a finite-state automaton from a GAP record file produced by
    the kbmag program.

    Arguments:
    ------------
    filename: the GAP record file containing automaton data.

    Return:
    --------
    An `FSA` object representing the automaton described by the
    file.

    """
    gap_record = gap_parse.load_record_file(filename)
    return _from_gap_record(gap_record)

def _from_gap_record(gap_record):
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
    """Load a finite-state automaton included with the automata
    subpackage.

    Parameters:
    -----------
    filename: the automaton file to use

    Return:
    ---------
    FSA object representing this automaton.

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
