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

def get_item_at(items, index):
    # SAFE example.
    # @requires 0 <= index
    # @requires index < len(items)
    # @ensures result == items[index]
    return items[index]

def has_valid_index(items, index):
    # SAFE example.
    # @ensures result == (0 <= index and index < len(items))
    return 0 <= index and index < len(items)
