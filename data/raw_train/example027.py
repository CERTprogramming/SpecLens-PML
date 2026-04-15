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

class BoundedCounter:
    # @invariant self.value >= 0
    # @invariant self.value <= self.limit

    def __init__(self, value, limit):
        # SAFE example.
        # @requires limit >= 0
        # @requires 0 <= value
        # @requires value <= limit
        # @ensures self.value == value
        # @ensures self.limit == limit
        self.value = value
        self.limit = limit

    def increment(self):
        # Intentionally incorrect implementation:
        # increases the counter even when it is already at the limit,
        # which may violate the class invariant.
        # @requires self.value < self.limit
        # @ensures self.value <= self.limit
        self.value += 1
        return self.value
