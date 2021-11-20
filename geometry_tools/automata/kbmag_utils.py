from collections import deque
from os import path
import subprocess


from . import gap_record_parse


KBMAG_PATH = "/home/teddy/math/tools/kbmag/bin"

def build_dict(v_list, labels, to_filter = []):
    v_dict = {}
    for i,neighbors in enumerate(v_list):
        n_dict = {}
        for label, vertex in zip(labels, neighbors):
            if vertex not in to_filter:
                if vertex not in n_dict:
                    n_dict[vertex] = []
                n_dict[vertex].append(label)

        v_dict[i+1] = n_dict
    return v_dict

def dict_to_dot(fsa_dict, name="FSA", newlines=True):
    if newlines:
        newline_char = "\n"
        tab_char = "\t"
    else:
        newline_char = ""
        tab_char = ""

    output = "digraph {} {{{}".format(name, newline_char)
    for v, neighbors in fsa_dict.items():
        if len(neighbors) == 0:
            output += '{};{}'.format(v, newline_char)
        else:
            for neighbor, label in neighbors.items():
                output += '{}{} -> {} [label="{}"];{}'.format(
                    tab_char, v, neighbor, label[0], newline_char
                )
    output += "}}{}".format(newline_char)
    return output

def run_kbmag(filename):
    autgroup = path.join(KBMAG_PATH, "autgroup")
    gpgeowa = path.join(KBMAG_PATH, "gpgeowa")


    subprocess.call([autgroup, filename])
    subprocess.call([gpgeowa, filename])

def load_fsa_dict(fsa_file):
    gap_dict = gap_record_parse.load_record_file(fsa_file)
    for fsa_dict in gap_dict.values():
        if fsa_dict["isFSA"] == "true":
            labels = fsa_dict["alphabet"]["names"]
            transitions = fsa_dict["table"]["transitions"]

            return build_dict(transitions, labels, to_filter = [0])

def follow_word(graph_dict, word, start_vertex):
    vertex = start_vertex
    for letter in word:
        vertex = graph_dict[vertex][letter]

    return vertex

def enumerate_fixed_length_paths(graph_dict, start_vertex, length):
    if length <= 0:
        yield ""
    else:
        for word in enumerate_fixed_length_paths(graph_dict, start_vertex, length - 1):
            vertex = follow_word(graph_dict, word, start_vertex)
            for label in graph_dict[vertex]:
                yield word + label

def enumerate_words(fsa, max_length):
    graph_dict = defaultdict(dict)
    for vertex, edge_dict in fsa.items():
        for neighbor, labels in edge_dict.items():
            graph_dict[vertex][labels[0]] = neighbor

    for i in range(max_length):
        for word in enumerate_fixed_length_paths(graph_dict, 1, i):
            yield word
