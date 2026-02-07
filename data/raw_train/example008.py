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

def positive(x):
    # Intentionally incorrect implementation:
    # returns x unchanged.
    # @requires True
    # @ensures result > 0
    return x

def abs_nonnegative(x):
    # SAFE example.
    # @requires True
    # @ensures result >= 0
    return abs(x)

def contradiction(x):
    # Intentionally impossible contract.
    # @requires True
    # @ensures result > x
    # @ensures result < x
    return x
