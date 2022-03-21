"""kbmag_utils.py: provide utility functions for working with the
kbmag command-line tools and their output.

"""

from collections import deque
from os import path
import subprocess

from . import gap_parse

KBMAG_PATH = "/home/teddy/math/tools/kbmag/bin"

def build_dict(transitions, labels, to_filter=[]):
    """Build a python dictionary from the data in the GAP record produced
    by kbmag to describe a finite-state automaton.

    Parameters:
    -----------

    transitions: list of tuples of length n, where n is the number of
    possible labels. If the ith entry of tuple j is k, then there is
    an edge from vertex j to vertex k with label i. Note that this
    implies that every vertex has outdegree len(labels).

    labels: ordered list of edge labels appearing in transitions.

    to_filter: vertices to discard when building the automaton
    (i.e. the "failure states" of the automaton).

    Return:
    -----------

    Dictionary describing the finite-state automaton, in the format
    expected by the fsa.FSA class.

    """
    v_dict = {}
    for i, neighbors in enumerate(transitions):
        n_dict = {}
        for label, vertex in zip(labels, neighbors):
            if vertex not in to_filter:
                n_dict[label] = vertex

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
