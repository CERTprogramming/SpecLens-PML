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

def power(x, n):
    # SAFE example.
    # @requires n >= 0
    # @ensures result >= 1
    return x ** n

def wrong_power(x, n):
    # Intentionally incorrect implementation:
    # uses subtraction instead of exponentiation, violating ensures.
    # @requires n >= 0
    # @ensures result == x ** n
    return x - n
