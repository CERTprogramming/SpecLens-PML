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

def append(values, x):
    # SAFE example.
    # @requires True
    # @ensures len(result) == len(values) + 1
    copy = list(values)
    copy.append(x)
    return copy

def index(values, i):
    # Intentionally incorrect implementation:
    # does not check bounds, may raise IndexError.
    # @requires True
    # @ensures result == values[i]
    return values[i]
