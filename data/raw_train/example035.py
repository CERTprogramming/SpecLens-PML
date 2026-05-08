# Example Python file annotated with PML contracts.
#
# This file is part of the SpecLens demo dataset:
# it contains examples using lightweight named snapshots.
#
# The goal is not full formal verification, but showcasing how pre-state
# values can be named and reused in readable postconditions.

def append_value(values, item):
    # SAFE example.
    # @snapshot old_len = len(values)
    # @ensures len(result) == old_len + 1
    return values + [item]

def append_value_risky(values, item):
    # Intentionally incorrect implementation:
    # returns the original list, violating the snapshot-based postcondition.
    # @snapshot old_len = len(values)
    # @ensures len(result) == old_len + 1
    return values

class SnapshotCounter:
    # @invariant self.value >= 0
    def __init__(self, value):
        # SAFE example.
        # @requires value >= 0
        # @ensures self.value == value
        self.value = value
    def add(self, amount):
        # SAFE method example.
        # @requires amount >= 0
        # @snapshot before = self.value
        # @ensures self.value == before + amount
        self.value += amount
        return self.value
