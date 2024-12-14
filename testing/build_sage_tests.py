import os
import re

from sage.repl import preparse

SCRIPT_DIR = os.path.dirname(__file__)
SAGE_DIR = "sage"
OUTPUT_DIR = os.path.join(SAGE_DIR, "preparsed")

TO_PREPARSE = [
    "test_projective.py",
    "test_automata.py",
    "test_hyperbolic.py",
    "test_representation.py",
    "test_drawing.py"
]

IMPORT_LINE = "from sage.all import RealNumber, Integer\n"

def change_common_filename(filename):
    prefix, ext = os.path.splitext(filename)
    return prefix + "_sage" + ext

def is_sage_test(filename):
    return re.match(r"test_.*\.py", os.path.basename(filename))

def change_sage_filename(filename):
    prefix, ext = os.path.splitext(filename)
    return prefix + "_preparsed" + ext

def preparse_common_tests():
    for filename in TO_PREPARSE:
        sagename = change_common_filename(filename)
        out_filename = os.path.join(SCRIPT_DIR, OUTPUT_DIR, sagename)
        abs_inpath = os.path.join(SCRIPT_DIR, filename)

        os.makedirs(os.path.dirname(out_filename), exist_ok=True)
        with open(out_filename, "w") as out_file:
            out_file.write(IMPORT_LINE)
            preparse.preparse_file_named_to_stream(abs_inpath, out_file)

def preparse_sage_tests():
    for filename in os.listdir(os.path.join(SCRIPT_DIR, SAGE_DIR)):

        if not is_sage_test(filename):
            continue

        preparsed_name = change_sage_filename(filename)
        out_filename = os.path.join(SCRIPT_DIR, OUTPUT_DIR, preparsed_name)
        abs_inpath = os.path.join(SCRIPT_DIR, SAGE_DIR, filename)

        os.makedirs(os.path.dirname(out_filename), exist_ok=True)
        with open(out_filename, "w") as out_file:
            out_file.write(IMPORT_LINE)
            preparse.preparse_file_named_to_stream(abs_inpath, out_file)

if __name__ == "__main__":
    preparse_common_tests()
    preparse_sage_tests()
