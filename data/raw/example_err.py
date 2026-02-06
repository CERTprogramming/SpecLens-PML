# Example Python file annotated with PML contracts.
# This file intentionally contains buggy implementations that may violate
# their specifications, providing negative samples for training and evaluation.

import math

def bad_sqrt(x):
    # @requires x >= 0
    # @ensures result >= 0
    # BUG: returns negative value for some inputs
    return -math.sqrt(x)

def unsafe_div(a, b):
    # @requires b != 0
    # @ensures result * b == a
    # BUG: uses true division, breaks the contract
    return a / b

class Counter:
    # @invariant self.value >= 0

    def __init__(self):
        self.value = 0

    def dec(self):
        # @ensures self.value >= 0
        # BUG: can violate the invariant
        self.value -= 1

class Wallet:
    # @invariant self.money >= 0

    def __init__(self, m):
        # @requires m >= 0
        self.money = m

    def spend(self, amount):
        # @requires amount > 0
        # @ensures self.money >= 0
        # BUG: no check, can go negative
        self.money -= amount
