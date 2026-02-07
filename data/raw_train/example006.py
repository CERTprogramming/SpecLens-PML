# Example Python file annotated with PML contracts.
#
# This file is part of the SpecLens demo dataset:
# it contains a mix of SAFE and intentionally RISKY functions.
#
# Some functions are deliberately implemented in a way that violates
# their declared postconditions or triggers runtime errors
# in order to provide negative examples for dataset generation and model training.
#
# The goal is not correctness of the implementation, but showcasing
# how contract violations can be detected and scored.

def first_char(s):
    # Intentionally incorrect implementation:
    # returns last char instead of first.
    # @requires len(s) > 0
    # @ensures result == s[0]
    return s[-1]

def uppercase(s):
    # SAFE example.
    # @requires len(s) > 0
    # @ensures result.isupper()
    return s.upper()
