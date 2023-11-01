import re

import pytest

from geometry_tools.automata import fsa
from geometry_tools.automata.fsa import FSAException

@pytest.fixture
def graphdict():
    vert_dict = {
        0 : {"a" : 1, "A": 2, "b":3, "B":4},
        1 : {"a" : 1, "b":3, "B":4},
        2 : {"A" : 2, "b":3, "B":4},
        3 : {"b" : 3, "a":1, "A":2},
        4 : {"B" : 4, "a":1, "A":2}
    }
    return vert_dict

@pytest.fixture
def graphdict_fsa(graphdict):
    return fsa.FSA(graphdict, start_vertices=[0],
                   graph_dict=True)

@pytest.fixture
def altdict():
    alt_dict = {
        0: {1: ["a"], 2: ["A"], 3: ["b"], 4:["B"]},
        1: {1: ["a"], 3: ["b"], 4:["B"]},
        2: {2: ["A"], 3: ["b"], 4:["B"]},
        3: {1: ["a"], 3: ["b"], 2:["A"]},
        4: {1: ["a"], 2: ["A"], 4:["B"]}
    }
    return alt_dict

@pytest.fixture
def alt_fsa(altdict):
    return fsa.FSA(altdict, start_vertices=[0],
                   graph_dict=False)

@pytest.fixture
def add_new_edge(graphdict_fsa):
    graphdict_fsa.add_edges([(4, 6, "b")])

def test_fsa_basics(graphdict_fsa):
    assert len(graphdict_fsa.vertices()) == 5
    assert len(list(graphdict_fsa.edges())) == 16
    assert (set(graphdict_fsa.neighbors_out(3)) ==
            {1, 2, 3})
    assert (set(graphdict_fsa.neighbors_in(3)) ==
            {0, 1, 2, 3})

    assert graphdict_fsa.edge_label(2, 3) == "b"

    with pytest.raises(ValueError):
        graphdict_fsa.edge_label(3, 4)

def test_alt_dict_fsa(alt_fsa):
    assert len(alt_fsa.vertices()) == 5
    assert len(list(alt_fsa.edges())) == 16
    assert (set(alt_fsa.neighbors_out(3)) ==
            {1, 2, 3})
    assert (set(alt_fsa.neighbors_in(3)) ==
            {0, 1, 2, 3})

    assert alt_fsa.edge_label(2, 3) == "b"

    with pytest.raises(ValueError):
        alt_fsa.edge_label(3, 4)

def test_add_vertices(graphdict_fsa):
    graphdict_fsa.add_vertices([5])

    assert 5 in graphdict_fsa.vertices()
    assert len(graphdict_fsa.neighbors_out(5)) == 0
    assert len(graphdict_fsa.neighbors_in(5)) == 0

def test_add_edge(graphdict_fsa):
    graphdict_fsa.add_edges([(1, 2, "c")])
    assert len(graphdict_fsa.neighbors_out(1)) == 4
    assert graphdict_fsa.edge_label(1,2) == "c"

def test_has_edge(graphdict_fsa):
    assert graphdict_fsa.has_edge(0, 4)
    assert not graphdict_fsa.has_edge(4, 0)
    assert not graphdict_fsa.has_edge(1, 2)

def test_add_edge_new_vertex(graphdict_fsa, add_new_edge):
    assert 6 in graphdict_fsa.vertices()
    assert graphdict_fsa.edge_label(4, 6) == "b"

def test_add_multiple_edges(graphdict_fsa):
    graphdict_fsa.add_edges([(1, 2, ["c", "d"]),
                             (3, 4, ["e"]),
                             (0, 1, ["E"])],
                            elist=True)

    assert len(list(graphdict_fsa.edges_out(1))) == 5
    assert len(list(graphdict_fsa.edges_in(2))) == 6
    assert graphdict_fsa.edge_label(3,4) == "e"

    with pytest.raises(ValueError):
        graphdict_fsa.edge_label(1,2)

    with pytest.raises(ValueError):
        graphdict_fsa.edge_label(0, 1)

def test_get_edges_with_labels(graphdict_fsa):
    assert set(graphdict_fsa.edges(with_labels=True)) == {
        (0, 1, "a"), (0, 2, "A"), (0, 3, "b"), (0, 4, "B"),
        (1, 1, "a"), (1, 3, "b"), (1, 4, "B"), (2, 2, "A"),
        (2, 3, "b"), (2, 4, "B"), (3, 1, "a"), (3, 2, "A"),
        (3, 3, "b"), (4, 1, "a"), (4, 2, "A"), (4, 4, "B")
    }

def test_get_edges_no_labels(graphdict_fsa):
    assert set(graphdict_fsa.edges(with_labels=False)) == {
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 1), (1, 3), (1, 4), (2, 2),
        (2, 3), (2, 4), (3, 1), (3, 2),
        (3, 3), (4, 1), (4, 2), (4, 4)
    }

def test_delete_vertex(graphdict_fsa):
    graphdict_fsa.delete_vertex(4)
    assert 4 not in list(graphdict_fsa.vertices())
    assert 4 not in graphdict_fsa.neighbors_out(0)
    assert 4 not in graphdict_fsa.neighbors_in(1)

def test_delete_vertices(graphdict_fsa):
    graphdict_fsa.delete_vertices([4,1])

    assert 4 not in list(graphdict_fsa.vertices())
    assert 4 not in graphdict_fsa.neighbors_out(0)
    assert 4 not in graphdict_fsa.neighbors_in(2)

    assert 1 not in list(graphdict_fsa.vertices())
    assert 1 not in graphdict_fsa.neighbors_out(0)
    assert 1 not in graphdict_fsa.neighbors_in(3)

def test_make_recurrent(graphdict_fsa, add_new_edge):
    assert 6 in graphdict_fsa.vertices()
    new_fsa = graphdict_fsa.recurrent(inplace=False)
    assert 6 in graphdict_fsa.vertices()
    assert len(list(graphdict_fsa.edges())) == 17
    assert graphdict_fsa.edge_label(4,6) == "b"

    assert set(new_fsa.vertices()) == {1, 2, 3, 4}
    assert len(list(new_fsa.edges())) == 12

def test_make_recurrent_inplace(graphdict_fsa, add_new_edge):
    assert 6 in graphdict_fsa.vertices()
    graphdict_fsa.recurrent(inplace=True)
    assert set(graphdict_fsa.vertices()) == {1, 2, 3, 4}
    assert len(list(graphdict_fsa.edges())) == 12

def test_remove_long_paths_with_ties(graphdict_fsa, add_new_edge):
    graphdict_fsa.add_edges([(6, 7, "c"),
                             (4, 7, "e"),
                             (3, 7, "f")])

    no_long_paths = graphdict_fsa.remove_long_paths(edge_ties=True)
    assert set(no_long_paths.edges(with_labels=True)) == {
        (0, 1, "a"), (0, 2, "A"), (0, 3, "b"), (0, 4, "B"),
        (4, 7, "e"), (4, 6, "b"), (3, 7, "f")
    }

def test_remove_long_paths_no_ties(graphdict_fsa, add_new_edge):
    graphdict_fsa.add_edges([(6, 7, "c"),
                             (4, 7, "e"),
                             (3, 7, "f")])

    no_long_paths = graphdict_fsa.remove_long_paths(edge_ties=False)
    assert set(no_long_paths.edges(with_labels=True)) <= {
        (0, 1, "a"), (0, 2, "A"), (0, 3, "b"), (0, 4, "B"),
        (4, 7, "e"), (4, 6, "b"), (3, 7, "f")
    }

    # boolean xor
    assert no_long_paths.has_edge(4, 7) != no_long_paths.has_edge(3, 7)

def test_follow_word(graphdict_fsa):
    assert graphdict_fsa.follow_word("abAAB") == 4
    with pytest.raises(FSAException):
        graphdict_fsa.follow_word("bAa")

def test_accepts(graphdict_fsa):
    assert graphdict_fsa.accepts("abaaaB")
    assert not graphdict_fsa.accepts("bbbababB")

def test_enumerate_fixed_length_paths(graphdict_fsa):
    result_0 = {"aaa", "aab", "aaB", "aba", "abA", "abb", "aBa", "aBA",
           "aBB", "AAA", "AAb", "AAB", "Aba", "AbA", "Abb", "ABa",
           "ABA", "ABB", "baa", "bab", "baB", "bAA", "bAb", "bAB",
           "bba", "bbA", "bbb", "Baa", "Bab", "BaB", "BAA", "BAb",
           "BAB", "BBa", "BBA", "BBB"}

    result_1 = { "aa", "ab", "aB", "ba", "bA", "bb", "Ba", "BA", "BB"}

    assert set(graphdict_fsa.enumerate_fixed_length_paths(
        3, start_vertex=0, with_states=False)
    ) == result_0

    assert set(graphdict_fsa.enumerate_fixed_length_paths(
        2, start_vertex=1, with_states=False)
    ) == result_1

def test_fixed_paths_with_states(graphdict_fsa):
    result_0 = {
        ("aa", 1), ("ab", 3), ("aB", 4), ("AA", 2), ("Ab", 3), ("AB", 4),
        ("ba", 1), ("bA", 2), ("bb", 3), ("Ba", 1), ("BA", 2), ("BB", 4)
    }
    result_1 = {
        ("aa", 1), ("ab", 3), ("aB", 4), ("ba", 1), ("bA", 2), ("bb", 3),
        ("Ba", 1), ("BA", 2), ("BB", 4)
    }

    assert set(graphdict_fsa.enumerate_fixed_length_paths(
        2, start_vertex=0, with_states=True)
    ) == result_0

    assert set(graphdict_fsa.enumerate_fixed_length_paths(
        2, start_vertex=1, with_states=True)
    ) == result_1

def test_enumerate_words(graphdict_fsa):
    result = {"aaa", "aab", "aaB", "aba", "abA", "abb", "aBa", "aBA",
              "aBB", "AAA", "AAb", "AAB", "Aba", "AbA", "Abb", "ABa",
              "ABA", "ABB", "baa", "bab", "baB", "bAA", "bAb", "bAB",
              "bba", "bbA", "bbb", "Baa", "Bab", "BaB", "BAA", "BAb",
              "BAB", "BBa", "BBA", "BBB", "aa", "ab", "aB", "AA", "Ab",
              "AB", "ba", "bA", "bb", "Ba", "BA", "BB",
              "a", "A", "b", "B", "A", ""}
    assert set(graphdict_fsa.enumerate_words(3, with_states=False)
    ) == result

def test_rename_generators(graphdict_fsa):
    rename_map = {"a": "c", "A":"C", "b":"e", "B":"B"}
    renamed = graphdict_fsa.rename_generators(rename_map, inplace=False)
    assert set(renamed.edges(with_labels=True)) == {
        (0, 1, "c"), (0, 2, "C"), (0, 3, "e"), (0, 4, "B"),
        (1, 1, "c"), (1, 3, "e"), (1, 4, "B"), (2, 2, "C"),
        (2, 3, "e"), (2, 4, "B"), (3, 1, "c"), (3, 2, "C"),
        (3, 3, "e"), (4, 1, "c"), (4, 2, "C"), (4, 4, "B")
    }

def test_rename_generators_inplace(graphdict_fsa):
    rename_map = {"a": "c", "A":"C", "b":"e", "B":"B"}
    renamed = graphdict_fsa.rename_generators(rename_map, inplace=True)
    assert set(graphdict_fsa.edges(with_labels=True)) == {
        (0, 1, "c"), (0, 2, "C"), (0, 3, "e"), (0, 4, "B"),
        (1, 1, "c"), (1, 3, "e"), (1, 4, "B"), (2, 2, "C"),
        (2, 3, "e"), (2, 4, "B"), (3, 1, "c"), (3, 2, "C"),
        (3, 3, "e"), (4, 1, "c"), (4, 2, "C"), (4, 4, "B")
    }

def test_automaton_multiple(graphdict_fsa):
    multiple = graphdict_fsa.automaton_multiple(3)
    assert (set(multiple.enumerate_fixed_length_paths(2)) ==
            set(graphdict_fsa.enumerate_fixed_length_paths(6)))

def test_load_kbmag():
    result = {1: {'d': 2, 'D': 3, 'c': 4, 'C': 5, 'b': 6, 'B': 7, 'a':
                  8, 'A': 9},
              2: {'d': 2, 'c': 4, 'C': 5, 'b': 6, 'B': 7, 'a': 8, 'A': 9},
              3: {'D': 3, 'c': 4, 'C': 5, 'b': 6, 'B': 7, 'a': 8, 'A':9},
              4: {'d': 2, 'D': 10, 'c': 4, 'b': 11, 'B': 7, 'a': 8, 'A': 9},
              5: {'d': 12, 'D': 13, 'C': 5, 'b': 6, 'B': 7, 'a': 8, 'A': 9},
              6: {'d': 2, 'D': 3, 'c': 4, 'C': 5, 'b': 6, 'a': 8, 'A': 14},
              7: {'d': 2, 'D': 3, 'c': 4, 'C': 15, 'B': 7, 'a': 8, 'A': 9},
              8: {'d': 16, 'D': 3, 'c': 4, 'C': 5, 'b': 6, 'B': 17, 'a': 8},
              9:{'d': 2, 'D': 3, 'c': 4, 'C': 5, 'b': 18, 'B': 19, 'A': 9},
              10: {'D': 3, 'c': 4, 'C': 5, 'b': 6, 'B': 7, 'a': 8, 'A': 20},
              11: {'d': 2, 'D': 3, 'c': 4, 'C': 5, 'b': 6, 'a': 8, 'A': 21},
              12: {'d': 2, 'c': 22, 'C': 5, 'b': 6, 'B': 7, 'a': 8, 'A': 9},
              13: {'D': 3, 'c': 23, 'C': 5, 'b': 6, 'B': 7, 'a': 8, 'A': 9},
              14: {'d': 2, 'D': 3, 'c': 4, 'C': 5, 'b': 18, 'B': 24, 'A': 9},
              15: {'d': 25, 'D': 13, 'C': 5, 'b': 6, 'B': 7, 'a': 8, 'A': 9},
              16: {'d': 2, 'c': 4, 'C': 26, 'b': 6, 'B': 7, 'a': 8, 'A': 9},
              17: {'d': 2, 'D': 3, 'c': 4, 'C': 27, 'B': 7, 'a': 8, 'A': 9},
              18: {'d': 2, 'D': 3, 'c': 4, 'C': 5, 'b': 6, 'a': 28, 'A': 14},
              19: {'d': 2, 'D': 3, 'c': 4, 'C': 15, 'B': 7, 'a': 29, 'A': 9},
              20: {'d': 2, 'D': 3, 'c': 4, 'C': 15, 'B': 19, 'A': 9},
              21: {'d': 2, 'D': 3, 'c': 4, 'C': 5, 'b': 18, 'A': 9},
              22: {'d': 2, 'D': 30, 'c': 4, 'b': 11, 'B': 7, 'a': 8, 'A': 9},
              23: {'d': 2, 'D': 10, 'c': 4, 'B': 7, 'a': 8, 'A': 9},
              24: {'d': 2, 'D': 3, 'c': 4, 'C': 15, 'B': 31, 'A': 9},
              25: {'d': 2, 'c': 32, 'C': 5, 'b': 6, 'B': 7, 'a': 8, 'A': 9},
              26: {'d': 12, 'C': 5, 'b': 6, 'B': 7, 'a': 8, 'A': 9},
              27: {'D': 13, 'C': 5, 'b': 6, 'B': 7, 'a': 8, 'A': 9},
              28: {'d': 16, 'D': 3, 'c': 4, 'C': 5, 'b': 6, 'a': 8},
              29: {'D': 3, 'c': 4, 'C': 5, 'b': 6, 'B': 17, 'a': 8},
              30: {'D': 3, 'c': 4, 'C': 5, 'b': 6, 'B': 33, 'a': 8},
              31: {'d': 2, 'D': 3, 'c': 4, 'C': 15, 'B': 7, 'a': 34, 'A': 9},
              32: {'d': 2, 'c': 4, 'b': 11, 'B': 7, 'a': 8, 'A': 9},
              33: {'d': 2, 'D': 3, 'c': 4, 'C': 35, 'B': 7, 'a': 8, 'A': 9},
              34: {'d': 36, 'D': 3, 'c': 4, 'C': 5, 'b': 6, 'B': 17, 'a': 8},
              35: {'d': 26, 'D': 13, 'C': 5, 'b': 6, 'B': 7, 'a': 8, 'A': 9},
              36: {'d': 2, 'c': 4, 'b': 6, 'B': 7, 'a': 8, 'A': 9}}
    oct_fsa = fsa.load_kbmag_file("local_files/octagon_surf.wa")
    assert oct_fsa.graph_dict == result

def test_list_builtins():
    # don't check the specific list, just check that it's a bunch of
    # files with .wa and .geowa extensions
    builtins_list = fsa.list_builtins()
    for name in builtins_list:
        assert re.match(r"(.*\.wa)|(.*\.geowa)", name)

def test_load_builtin():
    result = {1: {'a': 2, 'A': 3, 'b': 4, 'B': 5}, 2: {'a': 2, 'b': 4, 'B': 5},
              3: {'A': 3, 'b': 4, 'B': 5}, 4: {'a': 2, 'A': 3, 'b': 4},
              5: {'a': 2, 'A': 3, 'B': 5}}

    loaded = fsa.load_builtin("f2.wa")
    assert loaded.graph_dict == result

def test_free_automaton():
    free_automaton = fsa.free_automaton("abc")
    result = {
        "aa", "ab", "aB", "ac", "aC",
        "AA", "Ab", "AB", "Ac", "AC",
        "ba", "bA", "bb", "bc", "bC",
        "Ba", "BA", "BB", "Bc", "BC",
        "ca", "cA", "cb", "cB", "cc",
        "Ca", "CA", "Cb", "CB", "CC"
    }
    assert (set(free_automaton.enumerate_fixed_length_paths(2, with_states=False)) ==
            result)
