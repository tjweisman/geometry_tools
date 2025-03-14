import re
from collections import defaultdict
from itertools import product

SIMPLE_GENERATOR_NAMES = "abcdefghijklmnopqrstuvwxyz"

def invert_gen(generator):
    if generator.lower() == generator:
        return generator.upper()
    return generator.lower()

def formal_inverse(word, simple=True, inverse_map=invert_gen):
    joinchar = ""
    if not simple:
        joinchar = "*"
    return joinchar.join([inverse_map(g) for g in word[::-1]])

def asym_gens(generators):
    """Get an iterable of semigroup generators from an iterable of group generators.

    Given a sequence of lowercase/uppercase letters, return only the
    lowercase ones.

    Parameters
    ----------
    generators : iterable of strings
        Sequence of semigroup generators

    Yields
    ------
    gen : string
        the lowercase characters in `generators`

    """
    for gen in generators:
        if gen.lower() == gen:
            yield gen

def free_abelian_words(generators, length,
                       metric="sup", inverses=None):
    if inverses is None:
        inverses = [invert_gen(g) for g in generators]

    if metric == "sup":
        g_list = list(generators)
        n = len(g_list)
        for counts in product(range(-length, length + 1), repeat=n):
            word = ""
            for i, count in enumerate(counts):
                if count < 0:
                    word += inverses[i] * abs(count)
                elif count > 0:
                    word += g_list[i] * count
            yield word

def simplify_word(word, inverse_map=invert_gen,
                  as_string=True):
    simp = []
    for let in word:
        if len(simp) == 0 or let != inverse_map(simp[-1]):
            simp.append(let)
        else:
            simp = simp[:-1]

    if as_string:
        return "".join(simp)

    return simp

def simplify(zmod):
    return defaultdict(int, {simplify_word(word):zmod[word] for word in zmod})

def aug(zmod):
    return sum(zmod.values())

def commutator(w1, w2):
    return simplify_word(w1 + w2 + formal_inverse(w1) + formal_inverse(w2))

def act_left(word, zmod):
    prod = {word + zword:zmod[zword] for zword in zmod}
    return simplify(prod)

def act_right(zmod1, zmod2):
    mult = aug(zmod2)
    return defaultdict(int, {word:zmod1[word] * mult for word in zmod1})

def zmod_sum(z1, z2):
    z_sum = defaultdict(int, {word:coeff for word, coeff in z1.items()})
    for word in z2:
        z_sum[word] += z2[word]
    return z_sum

def fox_word_derivative(differential, word):
    if len(word) == 1:
        if word == differential:
            return defaultdict(int, {"":1})
        if word == formal_inverse(differential):
            return defaultdict(int, {word:-1})
        return defaultdict(int, {})

    return zmod_sum(
        fox_word_derivative(differential, word[0]),
        act_left(word[0], fox_word_derivative(differential, word[1:]))
    )
