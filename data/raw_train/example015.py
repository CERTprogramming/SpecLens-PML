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

def parse_nonnegative_int(text):
    # Intentionally incorrect implementation:
    # negates the parsed integer,
    # which violates the non-negativity postcondition.
    # @requires len(text) > 0
    # @ensures result >= 0
    return -int(text)

def is_valid_nonnegative_int(text):
    # SAFE example.
    # @requires len(text) > 0
    # @ensures result == (parse_nonnegative_int(text) >= 0)
    return parse_nonnegative_int(text) >= 0
