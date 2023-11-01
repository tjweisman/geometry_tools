#!/usr/local/bin/sage --python

import os
import re

from sage.repl import preparse

SCRIPT_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join("sage", "automatic")

TO_PREPARSE = [
    "test_projective.py"
]

IMPORT_LINE = "from sage.all import RealNumber, Integer\n"

def change_filename(filename):
    prefix, ext = os.path.splitext(filename)
    return prefix + "_sage" + ext

def preparse_files():
    for filename in TO_PREPARSE:
        sagename = change_filename(filename)
        out_filename = os.path.join(SCRIPT_DIR, OUTPUT_DIR, sagename)
        abs_inpath = os.path.join(SCRIPT_DIR, filename)

        os.makedirs(os.path.dirname(out_filename), exist_ok=True)
        with open(out_filename, "w") as out_file:
            out_file.write(IMPORT_LINE)
            preparse.preparse_file_named_to_stream(abs_inpath, out_file)

if __name__ == "__main__":
    preparse_files()
