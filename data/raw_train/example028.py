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

def reverse_text(text):
    # SAFE example.
    # @ensures len(result) == len(text)
    # @ensures result[::-1] == text
    return text[::-1]

def is_palindrome(text):
    # SAFE example.
    # @ensures result == (text == reverse_text(text))
    return text == reverse_text(text)
