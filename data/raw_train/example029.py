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

def last_item(items):
    # SAFE example.
    # @requires len(items) > 0
    # @ensures result == items[-1]
    return items[-1]

def remove_last_item(items):
    # Intentionally incorrect implementation:
    # removes the first item instead of the last one,
    # which violates the postcondition.
    # @requires len(items) > 0
    # @ensures len(result) == len(items) - 1
    # @ensures result == items[:-1]
    return items[1:]
