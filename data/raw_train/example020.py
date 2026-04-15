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

def sum_nonnegative(values):
    # SAFE example.
    # @requires all(v >= 0 for v in values)
    # @ensures result >= 0
    return sum(values)

def has_positive_sum(values):
    # SAFE example.
    # @requires all(v >= 0 for v in values)
    # @ensures result == (sum_nonnegative(values) > 0)
    return sum_nonnegative(values) > 0
