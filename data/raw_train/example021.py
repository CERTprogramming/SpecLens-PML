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

def absolute_difference(a, b):
    # Intentionally incorrect implementation:
    # returns the signed difference instead of the absolute one,
    # which may violate the non-negativity postcondition.
    # @ensures result >= 0
    return a - b

def same_value(a, b):
    # SAFE example.
    # @ensures result == (a == b)
    return a == b
