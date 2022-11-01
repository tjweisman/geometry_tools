"""coxeter_automaton

Code to generate a finite-state automaton accepting geodesic words in
Coxeter groups.

Code originally by Florian Stecker, with some small adaptations by
Teddy Weisman

"""

import numpy as np
import math
from copy import copy
from collections import deque, defaultdict
from . import fsa

class Root:
        def __init__(self, id, rank, depth = 0, v = None, neighbors = None):
                self.id = id
                self.rank = rank
                self.depth = depth
                if v:
                        self.v = v
                else:
                        self.v = [0] * rank
                if neighbors:
                        self.neighbors = neighbors
                else:
                        self.neighbors = [None] * rank

        def __copy__(self):
                return Root(self.id, self.rank, self.depth, self.v.copy(), self.neighbors.copy())

class Groupelement:
        def __init__(self, id, rank, word):
                self.id = id
                self.rank = rank
                self.word = word
                self.length = len(word)
                self.left = [None]*rank
                self.right = [None]*rank
                self.node = None
                self.lex_node = None
                self.inverse = None

# compute <alpha_k, beta> where alpha_k is one of the simple roots and beta any root
def form_gen_root(form, k, root):
        rank = len(form)
        return sum([root[i] * form[i][k] for i in range(rank)])

# compute beta - 2<alpha_k, beta>alpha_k, i.e. the reflection of beta along alpha_k
def apply_gen_to_root(form, k, root):
        root[k] -= 2*form_gen_root(form, k, root)

# find a sequence of generators to apply to obtain a negative root, from left to right
# "startwith" argument can be used to force the first entry
def find_word_to_negative(form, root_, startwith = None):
        rank = len(form)
        root = root_.copy()
        word = []
        while not next(filter(lambda x: x < -1e-6, root), None): # while root has no negative entry
                for k in range(rank):
                        if startwith and k != startwith:
                                continue
                        # avoiding 0 might be a problem for reducible groups?
                        f = form_gen_root(form, k, root)
                        if f > 1e-6:
                                apply_gen_to_root(form, k, root)
                                word.append(k)
                                break
                startwith = None
        return word

# use find_word_to_negative() to find the root, if we already have it
def find_root_from_vector(form, roots, vector):
        rank = len(form)
        for k in range(rank):
                word = find_word_to_negative(form, vector, startwith = k)

                if not word:
                        continue

                rootobj = roots[word.pop()]

                while len(word) > 0:
                        letter = word.pop()
                        if not rootobj.neighbors[letter]:
                                rootobj = None
                                break
                        else:
                                rootobj = rootobj.neighbors[letter]

                if rootobj:
                        return rootobj
        return None

def find_small_roots(form):
        rank = len(form)
        small_roots = []

        # the simple roots are just the standard basis vectors
        for i in range(rank):
                r = Root(i, rank)
                r.v[i] = 1
                r.depth = 1
                small_roots.append(r)

        # find the other small roots by applying generators to all existing roots
        # and using find_root_from_vector() to see if we already have it
        # then add it if it is a small root = was obtained via a short edge (form between -1 and 0)
        i = 0
        while i < len(small_roots):
                root = small_roots[i]
                for k in range(rank):
                        newroot = root.v.copy()
                        apply_gen_to_root(form, k, newroot)

                        rootobj = find_root_from_vector(form, small_roots, newroot)

                        if rootobj:
                                root.neighbors[k] = rootobj
                        else:
                                f = form_gen_root(form, k, root.v)
                                if f > -1 + 1e-6 and f < -1e-6:      # root is new and is a small root
                                        rootobj = Root(len(small_roots), rank, root.depth+1, newroot)
                                        small_roots.append(rootobj)
                                        root.neighbors[k] = rootobj
                i = i+1
        return small_roots


def apply_gen_to_node(small_roots, k, node, position, lex_reduced = False):
        # if we want to get the lex reduced langauge
        if lex_reduced:
                for j in range(k):
                        if small_roots[j].neighbors[k] and position == small_roots[j].neighbors[k].id:
                                return 1

        if position == k:
                return 1
        elif small_roots[position].neighbors[k]:
                swappos = small_roots[position].neighbors[k].id
                return node[swappos]
        else:
                return 0

def generate_automaton(small_roots, lex_reduced = False):
        nroots = len(small_roots)
        rank = small_roots[0].rank
        start = tuple([0]*nroots)
        todo = deque([start])
        nodes = {start: 0}
        levels = {start: 0}
        edges = []
        id = 1

        while todo:
                node = todo.pop()
                for k in range(rank):
                        if node[k] == 1:
                                continue
                        newnode = tuple(
                                apply_gen_to_node(small_roots, k, node, i, lex_reduced = lex_reduced)
                                for i in range(nroots))
                        if not newnode in nodes:
                                nodes[newnode] = id
                                levels[newnode] = levels[node]+1
                                todo.appendleft(newnode)
                                id += 1
                        edges.append((nodes[node], nodes[newnode], k))

        graph = defaultdict(dict)
        for (fr,to,gen) in edges:
                graph[fr][gen] = to

        return fsa.FSA(dict(graph), start_vertices=[0])


def enumerate_group(graph, graph_lex, max_len):
        rank = len(graph[0])
        group = [Groupelement(0, rank, tuple())]
        group[0].inverse = group[0]
        group[0].node = group[0].lex_node = 0

        i = 0
        size = 1
        while True:
                current = group[i]
                i+=1

                # break if current has the max length we have, as that's when we would start adding elements 1 longer
                if current.length >= max_len:
                        break

                for gen, new_lex_node in filter(lambda x: x[1], enumerate(graph_lex[current.lex_node])):
                        new_element = Groupelement(size, rank, current.word + (gen,))
                        new_element.lex_node = new_lex_node
                        new_element.node = graph[current.node][gen]
                        group.append(new_element)
                        size += 1

                        # w = w_1 t, w s = w_1
                        # right multiplication, if it decreases length
                        for k in range(rank):
                                if not graph[new_element.node][k]:
                                        word = list(new_element.word)
                                        longer_suffix = group[0]
                                        while len(word) > 0:
                                                letter = word.pop()
                                                shorter_suffix = longer_suffix
                                                longer_suffix = shorter_suffix.left[letter]

                                                # w = w_1 t w_2, w_2 s_k = t w_2
                                                # in the case word = [] longer_suffix could be None
                                                if len(word) == 0 or shorter_suffix.right[k] == longer_suffix:
                                                        # finish word
                                                        while len(word) > 0:
                                                                shorter_suffix = shorter_suffix.left[word.pop()]
                                                        new_element.right[k] = shorter_suffix
                                                        shorter_suffix.right[k] = new_element

                        # find inverse and left multiply
                        inverse = group[0]
                        word = list(new_element.word)
                        while len(word) > 0:
                                inverse = inverse.right[word.pop()]
                                if not inverse:
                                        break
                        if inverse:
                                new_element.inverse = inverse
                                inverse.inverse = new_element
                                for k in range(rank):
                                        if inverse.right[k]:
                                                other = inverse.right[k].inverse
                                                new_element.left[k] = other
                                                other.left[k] = new_element
                                        if new_element.right[k]:
                                                other = new_element.right[k].inverse
                                                inverse.left[k] = other
                                                other.left[k] = inverse
        return group

def word(w):
        return ''.join([chr(ord('a')+x) for x in w])

def generate_automaton_coxeter_matrix(coxeter_matrix, lex_reduced = False):
        form = [[-math.cos(math.pi/m) if m > 0 else -1 for m in row] for row in coxeter_matrix]
        rank = len(coxeter_matrix)
        small_roots = find_small_roots(form)
        return generate_automaton(small_roots, lex_reduced)

def even_graph(graph):
        rank = len(graph[0])
        result = []
        for node in graph:
                newnode = {}
                for i in range(rank):
                        for j in range(rank):
                                if node[i] and graph[node[i]][j]:
                                        newnode[(i,j)] = graph[node[i]][j]
                result.append(newnode)
        return result
