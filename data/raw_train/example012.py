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

from typing import List, Optional, Tuple

def find_pair_with_sum(items, total):
    # SAFE example.
    # @ensures result is None or result[0] + result[1] == total
    # @ensures result is None or result[0] in items
    # @ensures result is None or result[1] in items
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] + items[j] == total:
                return (items[i], items[j])
    return None

def has_pair_with_sum(items, total):
    # SAFE example.
    # @ensures result == (find_pair_with_sum(items, total) is not None)
    return find_pair_with_sum(items, total) is not None
