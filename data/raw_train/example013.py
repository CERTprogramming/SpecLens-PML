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

def verify_password_count(min_count, max_count, character, password):
    # SAFE example.
    # @requires min_count > 0
    # @requires max_count > 0
    # @requires min_count <= max_count
    # @requires len(character) == 1
    # @ensures len(password) != 0 or result == False
    return min_count <= password.count(character) <= max_count

def character_occurrences(character, password):
    # SAFE example.
    # @requires len(character) == 1
    # @ensures result >= 0
    return password.count(character)
