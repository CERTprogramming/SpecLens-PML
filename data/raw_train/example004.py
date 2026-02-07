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

def to_uppercase(s):
    # SAFE example.
    # @requires len(s) > 0
    # @ensures result.isupper()
    return s.upper()

def duplicate(values):
    # SAFE example.
    # @requires len(values) > 0
    # @ensures len(result) == 2 * len(values)
    return values + values

def truncate(s, n):
    # Intentionally incorrect implementation:
    # off-by-one bug.
    # @requires n >= 0
    # @ensures len(result) <= n
    return s[: n + 1]

def pop_element(values):
    # Intentionally incorrect implementation:
    # may raise IndexError on empty list.
    # @requires True
    # @ensures len(result) >= 0
    return values.pop()
