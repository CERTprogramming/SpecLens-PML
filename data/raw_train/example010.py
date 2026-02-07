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

def percentage(x, total):
    # Intentionally incorrect implementation:
    # uses floor division.
    # @requires total != 0
    # @ensures result * total == x * 100
    return (x * 100) // total

def ratio(x, total):
    # SAFE example.
    # @requires total != 0
    # @ensures result >= 0
    return (x * 1.0) / total
