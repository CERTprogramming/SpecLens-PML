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

def get_first_item(items):
    # Intentionally incorrect implementation:
    # returns the second item instead of the first one,
    # which violates the postcondition.
    # @requires len(items) > 0
    # @ensures result == items[0]
    return items[1]

def starts_with_first(items, value):
    # SAFE example.
    # @requires len(items) > 0
    # @ensures result == (items[0] == value)
    return items[0] == value
