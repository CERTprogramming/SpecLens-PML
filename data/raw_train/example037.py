# Example Python file annotated with PML contracts.
#
# This file is part of the SpecLens demo dataset:
# it contains examples adapted from Python-by-Contract corpus problems.
#
# The original corpus uses icontract annotations; these examples translate
# simple contract intent into SpecLens-PML comment-based syntax.

def first_item(values):
    # SAFE example.
    # @requires len(values) > 0
    # @ensures result == values[0]
    return values[0]

def first_item_risky(values):
    # Intentionally incorrect implementation:
    # returns the second item instead of the first one,
    # which violates the postcondition.
    # @requires len(values) > 1
    # @ensures result == values[0]
    return values[1]

def append_item(values, item):
    # SAFE example.
    # @snapshot old_len = len(values)
    # @ensures len(result) == old_len + 1
    # @ensures len(result) == old(len(values)) + 1
    # @ensures result[-1] == item
    return values + [item]

def remove_first(values):
    # SAFE example.
    # @requires len(values) > 0
    # @snapshot before = list(values)
    # @ensures len(result) == old(len(values)) - 1
    # @ensures result == before[1:]
    return values[1:]

def remove_first_risky(values):
    # Intentionally incorrect implementation:
    # removes the last item instead of the first one,
    # which violates the snapshot-based postcondition.
    # @requires len(values) > 0
    # @snapshot before = list(values)
    # @ensures len(result) == old(len(values)) - 1
    # @ensures result == before[1:]
    return values[:-1]
