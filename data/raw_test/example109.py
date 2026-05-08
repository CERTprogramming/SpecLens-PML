# Example Python file annotated with PML contracts.
#
# This file is part of the SpecLens demo dataset:
# it contains examples adapted from Python-by-Contract corpus problems.
#
# The original corpus uses icontract annotations; these examples translate
# simple contract intent into SpecLens-PML comment-based syntax.

class SeatInventory:
    # @invariant self.seats >= 0
    def __init__(self, value):
        # SAFE example.
        # @requires value >= 0
        # @ensures self.seats == value
        self.seats = value
    def reserve(self, amount):
        # SAFE example.
        # @requires 0 <= amount <= self.seats
        # @snapshot before = self.seats
        # @ensures self.seats == before - amount
        # @ensures self.seats == old(self.seats) - amount
        self.seats -= amount
        return self.seats
    def release_risky(self, amount):
        # Intentionally incorrect implementation:
        # releases one extra seat,
        # which violates the old-state postcondition.
        # @requires amount >= 0
        # @ensures self.seats == old(self.seats) + amount
        self.seats += amount + 1
        return self.seats
