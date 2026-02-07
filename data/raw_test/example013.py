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

def keys_count(d):
    # SAFE example.
    # @requires True
    # @ensures result == len(d)
    return len(d)

def get_value(d, key):
    # Intentionally incorrect implementation:
    # always returns None.
    # @requires key in d
    # @ensures result == d[key]
    return None

def update_value(d, key, value):
    # SAFE example.
    # @requires True
    # @ensures result[key] == value
    copy = dict(d)
    copy[key] = value
    return copy

def remove_key(d, key):
    # Intentionally incorrect implementation:
    # does not remove the key.
    # @requires key in d
    # @ensures key not in result
    return dict(d)
