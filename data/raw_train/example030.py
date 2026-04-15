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

    def __init__(self, balance):
        # SAFE example.
        # @requires balance >= 0
        # @ensures self.balance == balance
        self.balance = balance

    def deposit(self, amount):
        # SAFE example.
        # @requires amount >= 0
        # @ensures self.balance >= 0
        self.balance += amount
        return self.balance

    def withdraw(self, amount):
        # Intentionally incorrect implementation:
        # subtracts the amount without checking whether enough balance is available,
        # which may violate the class invariant.
        # @requires amount >= 0
        # @requires amount <= self.balance
        # @ensures self.balance >= 0
        self.balance -= amount
        return self.balance
