import math


def div(a, b):
    # @requires b != 0
    # @ensures result * b == a
    return a // b


def sqrt(x):
    # @requires x >= 0
    # @ensures result >= 0
    return math.sqrt(x)


def mean(values):
    # @requires len(values) > 0
    # @ensures result >= min(values)
    # @ensures result <= max(values)
    return sum(values) / len(values)


def clamp(x, lo, hi):
    # @requires lo <= hi
    # @ensures result >= lo
    # @ensures result <= hi
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
