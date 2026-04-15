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

def count_increases(values):
    # SAFE example.
    # @ensures result >= 0
    # @ensures result <= len(values)
    increases = 0
    for i in range(1, len(values)):
        if values[i] > values[i - 1]:
            increases += 1
    return increases

def has_increase(values):
    # SAFE example.
    # @ensures result == (count_increases(values) > 0)
    return count_increases(values) > 0
