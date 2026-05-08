# Example Python file annotated with PML contracts.
#
# This file is part of the SpecLens demo dataset:
# it contains examples adapted from Python-by-Contract corpus problems.
#
# The original corpus uses icontract annotations; these examples translate
# simple contract intent into SpecLens-PML comment-based syntax.

class CountingBag:
    # @invariant self.count >= 0
    def __init__(self, value):
        # SAFE example.
        # @requires value >= 0
        # @ensures self.count == value
        self.count = value
    def push(self, amount):
        # SAFE example.
        # @requires amount >= 0
        # @snapshot before = self.count
        # @ensures self.count == before + amount
        self.count += amount
        return self.count
    def pop(self):
        # SAFE example.
        # @requires self.count > 0
        # @ensures self.count == old(self.count) - 1
        self.count -= 1
        return self.count
    def pop_risky(self):
        # Intentionally incorrect implementation:
        # removes two stored items instead of one,
        # which violates the old-state postcondition.
        # @requires self.count > 1
        # @ensures self.count == old(self.count) - 1
        self.count -= 2
        return self.count
