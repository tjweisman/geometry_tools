from collections import deque, defaultdict
import copy

class FSA:
    def __init__(self, vert_dict):
        self._out_dict = copy.deepcopy(vert_dict)
        self._in_dict = self.in_dict()

    def in_dict(self):
        in_dict = {v:{} for v in self._out_dict}
        for v, neighbors_out in self._out_dict.items():
            for w, labels in neighbors_out.items():
                in_dict[w][v] = labels
        return in_dict

    def edges_out(self, vertex):
        for w, labels in self._out_dict[vertex].items():
            for label in labels:
                yield (vertex, w, label)

    def edges_in(self, vertex):
        for w, labels in self._in_dict[vertex].items():
            for label in labels:
                yield (w, vertex, label)

    def neighbors_out(self, vertex):
        return self._out_dict[vertex].keys()

    def neighbors_in(self, vertex):
        return self._in_dict[vertex].keys()

    def edge_label(self, tail, head):
        if len(self._out_dict[tail][head]) == 1:
            return self._out_dict[tail][head][0]
        else:
            raise Exception("ambiguous: there is not exactly"
                            " one edge between these vertices!" )

    def edge_labels(self, tail, head):
        return self._out_dict[tail][head]

    def add_vertices(self, vertices):
        for v in vertices:
            if v not in self._out_dict:
                self._out_dict[v] = {}
                self._in_dict[v] = {}

    def add_edges(self, edges, elist=False):
        for e in edges:
            tail, head, label = e

            if head not in self._out_dict[tail]:
                self._out_dict[tail][head] = []
                self._in_dict[head][tail] = []

            if elist:
                self._out_dict[tail][head] += label
                self._in_dict[head][tail] += label
            else:
                self._out_dict[tail][head].append(label)
                self._in_dict[head][tail].append(label)

    def delete_vertices(self, vertices):
        for v in vertices:
            self.delete_vertex(v)

    def delete_vertex(self, vertex):
        for w in self.neighbors_out(vertex):
            self._in_dict[w].pop(vertex)
        for w in self.neighbors_in(vertex):
            self._out_dict[w].pop(vertex)

        self._out_dict.pop(vertex)
        self._in_dict.pop(vertex)

    def vertices(self):
        return self._out_dict.keys()

    def edges(self):
        for v, nbrs in self._out_dict.items():
            for w in nbrs:
                yield (v,w)

    def make_recurrent(self):
        still_pruning = True
        while still_pruning:
            still_pruning = False
            for v in self._out_dict.keys():
                if (len(self._out_dict[v]) == 0 or
                    len(self._in_dict[v]) == 0):
                    self.delete_vertex(v)
                    still_pruning = True

def remove_long_paths(G, root, edge_ties=True, return_distances = False):
    H = FSA({})
    H.add_vertices(G.vertices())

    #Dijkstra's algorithm!
    distance = {}
    marked = {v:False for v in G.vertices()}

    marked[root] = True
    distance[root] = 0
    vertex_queue = deque([root])
    while len(vertex_queue) > 0:
        v = vertex_queue.popleft()

        to_visit = [w for w in G.neighbors_out(v) if not marked[w]]
        for w in to_visit:
            marked[w] = True
            distance[w] = distance[v] + 1

        short_nbrs = to_visit
        if edge_ties:
            short_nbrs = [w for w in G.neighbors_out(v)
                           if distance[w] == distance[v] + 1]

        H.add_edges([(v,w,G.edge_labels(v,w)) for w in short_nbrs],
                    elist= True)
        vertex_queue.extend(to_visit)


    if return_distances:
        return H, distance
    else:
        return H

def follow_word(graph_dict, word, start_vertex):
    vertex = start_vertex
    for letter in word:
        vertex = graph_dict[vertex][letter]

    return vertex

def enumerate_fixed_length_paths(graph_dict, start_vertex, length,
                                 with_states=False):
    if length <= 0:
        if with_states:
            yield ("", 1)
        else:
            yield ""
    else:
        for word in enumerate_fixed_length_paths(graph_dict, start_vertex, length - 1):
            vertex = follow_word(graph_dict, word, start_vertex)
            for label in graph_dict[vertex]:
                if with_states:
                    yield (word + label, graph_dict[vertex][label])
                else:
                    yield word + label

def enumerate_words(fsa, max_length, with_states=False):
    graph_dict = defaultdict(dict)
    for vertex, edge_dict in fsa.items():
        for neighbor, labels in edge_dict.items():
            graph_dict[vertex][labels[0]] = neighbor

    for i in range(max_length):
        for word in enumerate_fixed_length_paths(graph_dict, 1, i,
                                                 with_states):
            yield word
