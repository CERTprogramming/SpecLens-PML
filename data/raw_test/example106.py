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

class Stack:
    # @invariant len(self.items) >= 0

    def __init__(self):
        # SAFE example.
        # @ensures len(self.items) == 0
        self.items = []

    def push(self, item):
        # SAFE example.
        # @ensures len(self.items) >= 1
        self.items.append(item)
        return len(self.items)

    def pop_item(self):
        # Intentionally incorrect implementation:
        # removes and returns the first item instead of the last one,
        # which violates the stack discipline.
        # @requires len(self.items) > 0
        # @ensures len(self.items) >= 0
        return self.items.pop(0)
