# Example Python file annotated with PML contracts.
#
# This file is part of the SpecLens demo dataset:
# it contains examples adapted from Python-by-Contract corpus problems.
#
# The original corpus uses icontract annotations; these examples translate
# simple contract intent into SpecLens-PML comment-based syntax.

def next_departure(bus_id, min_time):
    # SAFE example.
    # @requires bus_id > 0
    # @requires min_time >= 0
    # @ensures result >= min_time
    # @ensures result < min_time + bus_id
    # @ensures result % bus_id == 0
    missed = min_time % bus_id
    if missed == 0:
        return min_time
    return min_time - missed + bus_id

def next_departure_risky(bus_id, min_time):
    # Intentionally incorrect implementation:
    # always skips one full period,
    # which violates the upper-bound postcondition when min_time is aligned.
    # @requires bus_id > 0
    # @requires min_time >= 0
    # @ensures result >= min_time
    # @ensures result < min_time + bus_id
    # @ensures result % bus_id == 0
    return min_time + bus_id

def lower_half(first, last):
    # SAFE example.
    # @requires 0 <= first < last
    # @requires (last - first + 1) % 2 == 0
    # @ensures result[0] == first
    # @ensures result[1] < last
    # @ensures first <= result[0] <= result[1] <= last
    half = (last - first + 1) // 2
    return (first, last - half)

def upper_half_risky(first, last):
    # Intentionally incorrect implementation:
    # advances the lower bound one step too far,
    # which can move the result outside the original range.
    # @requires 0 <= first < last
    # @requires (last - first + 1) % 2 == 0
    # @ensures result[0] > first
    # @ensures result[1] == last
    # @ensures first <= result[0] <= result[1] <= last
    half = (last - first + 1) // 2
    return (first + half + 1, last)
