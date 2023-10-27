import re
from collections import defaultdict

def invert_gen(generator):
    if generator.lower() == generator:
        return generator.upper()
    return generator.lower()

def formal_inverse(word):
    return "".join([invert_gen(g) for g in word[::-1]])

def simplify_word(word):
    simp = ""
    for let in word:
        if len(simp) == 0 or let != invert_gen(simp[-1]):
            simp += let
        else:
            simp = simp[:-1]
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
