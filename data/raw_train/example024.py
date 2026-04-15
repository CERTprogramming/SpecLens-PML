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

def maximum_value(values):
    # SAFE example.
    # @requires len(values) > 0
    # @ensures result in values
    # @ensures all(result >= v for v in values)
    return max(values)

def has_single_value(values):
    # SAFE example.
    # @ensures result == (len(values) == 1)
    return len(values) == 1
