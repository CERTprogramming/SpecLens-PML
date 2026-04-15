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

def safe_divide(total, count):
    # SAFE example.
    # @requires count != 0
    # @ensures result * count == total
    return total / count

def floor_divide(total, count):
    # Intentionally incorrect implementation:
    # uses floor division instead of true division,
    # which violates the postcondition.
    # @requires count != 0
    # @ensures result * count == total
    return total // count
