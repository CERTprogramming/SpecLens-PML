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

def first_character(text):
    # Intentionally incorrect implementation:
    # returns the last character instead of the first one,
    # which violates the postcondition.
    # @requires len(text) > 0
    # @ensures result == text[0]
    return text[-1]

def is_empty(text):
    # SAFE example.
    # @ensures result == (len(text) == 0)
    return len(text) == 0
