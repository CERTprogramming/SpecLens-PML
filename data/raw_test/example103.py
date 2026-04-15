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

def count_occurrences(items, value):
    # SAFE example.
    # @ensures result >= 0
    return items.count(value)

def contains_value(items, value):
    # SAFE example.
    # @ensures result == (count_occurrences(items, value) > 0)
    return count_occurrences(items, value) > 0
