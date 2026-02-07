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

def reciprocal(x):
    # Intentionally incorrect implementation:
    # may raise ZeroDivisionError when x == 0.
    # @requires True
    # @ensures result > 0
    return 1 / x

def abs_value(x):
    # Intentionally incorrect implementation:
    # violates the postcondition for negative inputs.
    # @requires True
    # @ensures result >= 0
    return x

def maximum(a, b):
    # SAFE example.
    # @requires True
    # @ensures result >= a
    # @ensures result >= b
    return a if a >= b else b

def impossible(x):
    # Intentionally impossible contract.
    # @requires True
    # @ensures result < x
    # @ensures result > x
    return x
