# Example Python file annotated with PML contracts.
#
# This file is part of the SpecLens demo dataset:
# it contains examples adapted from Python-by-Contract corpus problems.
#
# The original corpus uses icontract annotations; these examples translate
# simple contract intent into SpecLens-PML comment-based syntax.

def has_character(text, name):
    # SAFE example.
    # @requires len(name) == 1
    # @ensures result == (name in text)
    return name in text

def has_character_risky(text, name):
    # Intentionally incorrect implementation:
    # returns the negated membership result,
    # which violates the membership postcondition.
    # @requires len(name) == 1
    # @ensures result == (name in text)
    return not (name in text)

def first_character(text):
    # SAFE example.
    # @requires len(text) > 0
    # @ensures result == text[0]
    return text[0]

def first_character_risky(text):
    # Intentionally incorrect implementation:
    # returns the last character instead of the first one,
    # which violates the postcondition.
    # @requires len(text) > 0
    # @ensures result == text[0]
    return text[-1]

def password_count_ok(min_count, max_count, name, text):
    # SAFE example.
    # @requires min_count >= 0
    # @requires max_count >= min_count
    # @requires len(name) == 1
    # @ensures result == (min_count <= text.count(name) <= max_count)
    return min_count <= text.count(name) <= max_count
