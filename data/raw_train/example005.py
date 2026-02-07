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

def square(x):
    # SAFE example.
    # @requires True
    # @ensures result >= 0
    return x * x

def decrement(x):
    # Intentionally incorrect implementation:
    # violates ensures because result < x.
    # @requires True
    # @ensures result >= x
    return x - 1

def add(a, b):
    # SAFE example.
    # @requires True
    # @ensures result == a + b
    return a + b
