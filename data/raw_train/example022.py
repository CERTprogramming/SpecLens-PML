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

def safe_average(values):
    # SAFE example.
    # @requires len(values) > 0
    # @ensures min(values) <= result
    # @ensures result <= max(values)
    return sum(values) / len(values)

def has_average(values):
    # SAFE example.
    # @ensures result == (len(values) > 0)
    return len(values) > 0
