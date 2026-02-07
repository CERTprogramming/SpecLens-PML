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

class Account:
    # @invariant self.balance >= 0

    def __init__(self, b):
        # SAFE constructor.
        # @requires b >= 0
        self.balance = b

    def deposit(self, amount):
        # SAFE example.
        # @requires amount > 0
        # @ensures self.balance >= 0
        self.balance += amount

    def withdraw(self, amount):
        # Intentionally incorrect implementation:
        # may violate the invariant by allowing negative balance.
        # @requires amount > 0
        # @ensures self.balance >= 0
        self.balance -= amount

    def transfer_to(self, other, amount):
        # SAFE example.
        # @requires amount > 0
        # @ensures self.balance >= 0
        # @ensures other.balance >= 0
        if amount > self.balance:
            amount = self.balance
        self.balance -= amount
        other.balance += amount
