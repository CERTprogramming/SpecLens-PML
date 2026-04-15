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

def clip_to_unit_interval(value):
    # SAFE example.
    # @ensures 0.0 <= result
    # @ensures result <= 1.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value

def is_unit_interval_value(value):
    # SAFE example.
    # @ensures result == (0.0 <= value and value <= 1.0)
    return 0.0 <= value and value <= 1.0
