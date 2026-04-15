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

def prefix(text, length):
    # SAFE example.
    # @requires 0 <= length
    # @requires length <= len(text)
    # @ensures len(result) == length
    # @ensures result == text[:length]
    return text[:length]

def suffix(text, length):
    # SAFE example.
    # @requires 0 <= length
    # @requires length <= len(text)
    # @ensures len(result) == length
    # @ensures result == text[len(text) - length:]
    return text[len(text) - length:]
