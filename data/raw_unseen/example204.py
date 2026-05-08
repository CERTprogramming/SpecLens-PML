# Example Python file annotated with PML contracts.
#
# This file is part of the SpecLens demo dataset:
# it contains examples adapted from Python-by-Contract corpus problems.
#
# The original corpus uses icontract annotations; these examples translate
# simple contract intent into SpecLens-PML comment-based syntax.

def adapter_product(delta_ones, delta_threes):
    # SAFE example.
    # @requires delta_ones >= 0
    # @requires delta_threes >= 0
    # @ensures result >= 0
    # @ensures result == delta_ones * delta_threes
    return delta_ones * delta_threes

def adapter_product_risky(delta_ones, delta_threes):
    # Intentionally incorrect implementation:
    # adds the histogram bins instead of multiplying them,
    # which violates the product postcondition.
    # @requires delta_ones > 1
    # @requires delta_threes > 1
    # @ensures result == delta_ones * delta_threes
    return delta_ones + delta_threes

def take_prefix(values, index):
    # SAFE example.
    # @requires 0 <= index <= len(values)
    # @snapshot old_len = len(values)
    # @ensures len(result) <= old_len
    # @ensures len(result) <= old(len(values))
    # @ensures result == values[:index]
    return values[:index]

def take_prefix_risky(values, index):
    # Intentionally incorrect implementation:
    # drops one requested prefix item,
    # which violates the snapshot-based postcondition.
    # @requires 0 < index <= len(values)
    # @snapshot before = list(values)
    # @ensures result == before[:index]
    return values[:index - 1]

def replace_first(values, item):
    # SAFE example.
    # @requires len(values) > 0
    # @snapshot before = list(values)
    # @ensures len(result) == old(len(values))
    # @ensures result[0] == item
    # @ensures result[1:] == before[1:]
    return [item] + values[1:]
