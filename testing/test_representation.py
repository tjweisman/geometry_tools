import pytest
import numpy as np

from geometry_tools.representation import Representation

@pytest.fixture
def cyclic_representation():
    rep = Representation()
    rep["a"] = np.array([
        [2.0, 0.0],
        [0.0, 0.5]
    ])
    return rep

@pytest.fixture
def free_representation(cyclic_representation):
    rot = np.array([
        [np.cos(np.pi/4), -np.sin(np.pi/4)],
        [np.sin(np.pi/4), np.cos(np.pi/4)]
    ])
    cyclic_representation["b"] = rot @ cyclic_representation["a"] @ np.linalg.inv(rot)
    return cyclic_representation

@pytest.fixture
def free_word_list():
    words = {
        "", "a", "A", "b", "B", "aa", "ab", "aB", "AA", "Ab", "AB",
        "ba", "bA", "bb", "Ba", "BA", "BB"
    }
    return words

@pytest.fixture
def conjugator_2x2():
    return np.array([
        [1.4, -0.66],
        [0.76, 11.5]
    ])

@pytest.fixture
def longname_rep():
    rep = Representation()
    rep["word1"] = np.array([
        [1.0, 1.0],
        [0.0, 1.0]
    ])
    rep["word2"] = np.array([
        [1.0, 0.0],
        [1.0, 1.0]
    ])
    return rep


def test_representation(cyclic_representation):
    rep = cyclic_representation
    assert np.allclose(
        rep["a"],
        np.array([
            [2.0, 0.0],
            [0.0, 0.5]
        ])
    )
    assert np.allclose(
        rep["aa"],
        rep["a"] @ rep["a"]
    )
    assert np.allclose(
        rep["aA"],
        np.identity(2)
    )

def test_asym_gens(free_representation):
    assert set(free_representation.asym_gens()) == {"a", "b"}

def test_free_words(free_representation, free_word_list):
    words = set(free_representation.free_words_less_than(3))
    assert words == free_word_list

def test_free_elements(free_representation, free_word_list):
    test_vals = {
        word: free_representation[word]
        for word in free_word_list
    }
    vals = {
        word:image
        for image, word in zip(*free_representation.freely_reduced_elements(
                2, with_words=True)
        )
    }
    assert set(vals.keys()) == set(test_vals.keys())
    for word in vals:
        assert np.allclose(vals[word], test_vals[word])

def test_conjugate(free_representation, conjugator_2x2):
    word = "abbbaBB"
    conjugate_rep = free_representation.conjugate(conjugator_2x2)
    assert np.allclose(
        conjugate_rep[word],
        np.linalg.inv(conjugator_2x2) @ free_representation[word] @ conjugator_2x2
    )

def test_longname_rep(longname_rep):
    commutator = longname_rep[["word1", "word2", "WORD1", "WORD2"]]

    g1 = longname_rep[["word1"]]
    g2 = longname_rep[["word2"]]

    assert np.allclose(commutator,
                       g1 @ g2 @ np.linalg.inv(g1) @ np.linalg.inv(g2)
    )

def test_compose(free_representation):
    # compose with identity
    composed_rep = free_representation.compose(lambda x: x)

    word = "abAABB"

    assert np.allclose(free_representation[word],
                       composed_rep[word])
