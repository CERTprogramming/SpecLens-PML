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

def strip_spaces(s):
    # SAFE example.
    # @requires len(s) > 0
    # @ensures len(result) <= len(s)
    return s.strip()

def reverse_string(s):
    # Intentionally incorrect implementation:
    # returns original string instead of reversed.
    # @requires len(s) > 0
    # @ensures result == s[::-1]
    return s

def repeat(s, n):
    # SAFE example.
    # @requires n >= 0
    # @ensures len(result) == n * len(s)
    return s * n
