class Account:
    # @invariant self.balance >= 0

    def __init__(self, b):
        # @requires b >= 0
        self.balance = b

    def deposit(self, amount):
        # @requires amount > 0
        # @ensures self.balance >= 0
        self.balance += amount

    def withdraw(self, amount):
        # @requires amount > 0
        # @ensures self.balance >= 0
        if amount > self.balance:
            amount = self.balance
        self.balance -= amount

    def transfer_to(self, other, amount):
        # @requires amount > 0
        # @ensures self.balance >= 0
        # @ensures other.balance >= 0
        if amount > self.balance:
            amount = self.balance
        self.balance -= amount
        other.balance += amount


def safe_div(a, b):
    # @requires b != 0
    # @ensures result * b == a
    return a // b
