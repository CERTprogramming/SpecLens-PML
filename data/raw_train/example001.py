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

import math

def div(a, b):
    # Intentionally incorrect implementation:
    # uses integer (floor) division, which violates the postcondition.
    # @requires b != 0
    # @ensures result * b == a
    return a // b

def sqrt(x):
    # SAFE example.
    # @requires x >= 0
    # @ensures result >= 0
    return math.sqrt(x)

def mean(values):
    # SAFE example.
    # @requires len(values) > 0
    # @ensures result >= min(values)
    # @ensures result <= max(values)
    return sum(values) / len(values)

def clamp(x, lo, hi):
    # SAFE example.
    # @requires lo <= hi
    # @ensures result >= lo
    # @ensures result <= hi
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
