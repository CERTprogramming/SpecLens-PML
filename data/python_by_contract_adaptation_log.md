# Python-by-Contract Adaptation Log

This batch adapts simple contract ideas from the Python-by-Contract corpus into
SpecLens-PML comment contracts. The source corpus uses `icontract`; the examples
added here avoid runtime dependencies on the corpus and keep only semantics that
fit the current parser and dynamic labeler.

## Adapted Themes

- AoC 2020 day 1: pair/list membership and simple collection postconditions.
- AoC 2020 day 5: binary range narrowing and seat-like indexed arithmetic.
- AoC 2020 day 10: adapter histogram arithmetic and collection size contracts.
- AoC 2020 day 13: next-departure arithmetic bounds.
- AoC 2020 day 2: character-count password checks.
- ETHZ exercise 06 linked-list task: add/remove state changes with snapshots.

## Skipped Or Simplified

- Regex-heavy parsers were skipped because the current examples should avoid
  complex parsing contracts and framework-specific regex semantics.
- Visitor trees, AST interpreters, and recursive parsers were skipped because
  they require interprocedural behavior and heavy object models.
- Custom `DBC` subclasses and collection wrapper types were simplified to plain
  Python values so `demo.py` can import and execute the examples directly.
- Advanced `icontract.snapshot` cases over cursors or nested linked nodes were
  simplified to integer counts or copied lists through `@snapshot`.
- Floating-point tolerance examples were skipped because they need numerical
  semantics beyond the current lightweight contract evaluator.
