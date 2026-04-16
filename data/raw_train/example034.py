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

class CounterWithOld:
    # @invariant self.value >= 0

    def __init__(self, value):
        # SAFE example.
        # @requires value >= 0
        # @ensures self.value == value
        self.value = value

    def increment(self):
        # SAFE example.
        # @ensures self.value == old(self.value) + 1
        self.value += 1
        return self.value

    def decrement(self):
        # Intentionally incorrect implementation:
        # subtracts 2 instead of 1,
        # which violates the postcondition based on old(self.value).
        # @requires self.value > 0
        # @ensures self.value == old(self.value) - 1
        self.value -= 2
        return self.value
