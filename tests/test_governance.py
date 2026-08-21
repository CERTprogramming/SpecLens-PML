"""Focused regression tests for the concept-aware governance layer."""

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from governance.concepts import extract_concept_states
from governance.control import append_control_event, apply_control, load_policies
from governance.review import _load_control_event, append_review


ROOT = Path(__file__).resolve().parents[1]


class GovernanceControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policies = load_policies(ROOT / "governance" / "policies.yaml")

    def test_p001_escalates_medium_to_high_without_changing_score(self) -> None:
        states = extract_concept_states(
            {"has_subscript": 1, "has_missing_requires": 1},
            {"INDEX_ACCESS", "MISSING_PRECONDITION_COVERAGE"},
        )
        decision = apply_control(
            score=0.53,
            original_level="MEDIUM",
            concept_states=states,
            policy_document=self.policies,
            low_threshold=0.20,
            high_threshold=0.60,
        )

        self.assertEqual(decision.model_score, 0.53)
        self.assertEqual(decision.original_level, "MEDIUM")
        self.assertEqual(decision.controlled_high_threshold, 0.50)
        self.assertEqual(decision.controlled_level, "HIGH")
        self.assertTrue(decision.decision_changed)
        self.assertTrue(decision.review_required)
        self.assertEqual(decision.matched_policy_ids, ("P001",))

    def test_non_matching_policy_preserves_baseline(self) -> None:
        decision = apply_control(
            score=0.53,
            original_level="MEDIUM",
            concept_states={
                "INDEX_ACCESS": "present",
                "MISSING_PRECONDITION_COVERAGE": "absent",
            },
            policy_document=self.policies,
            low_threshold=0.20,
            high_threshold=0.60,
        )
        self.assertEqual(decision.controlled_level, "MEDIUM")
        self.assertFalse(decision.decision_changed)
        self.assertEqual(decision.matched_policy_ids, ())

    def test_less_conservative_policy_is_rejected(self) -> None:
        unsafe = {
            "version": 1,
            "policies": [
                {
                    "id": "PX",
                    "enabled": True,
                    "when": {"INDEX_ACCESS": "present"},
                    "high_risk_threshold": 0.75,
                }
            ],
        }
        with self.assertRaises(ValueError):
            apply_control(
                score=0.53,
                original_level="MEDIUM",
                concept_states={"INDEX_ACCESS": "present"},
                policy_document=unsafe,
                low_threshold=0.20,
                high_threshold=0.60,
            )

    def test_control_and_human_review_are_append_only(self) -> None:
        decision = apply_control(
            score=0.53,
            original_level="MEDIUM",
            concept_states={
                "INDEX_ACCESS": "present",
                "MISSING_PRECONDITION_COVERAGE": "present",
            },
            policy_document=self.policies,
            low_threshold=0.20,
            high_threshold=0.60,
        )

        with TemporaryDirectory() as tmp:
            log = Path(tmp) / "control_events.jsonl"
            event_id = append_control_event(
                decision,
                function_name="get_item",
                source_file="example.py",
                line=10,
                concept_states={
                    "INDEX_ACCESS": "present",
                    "MISSING_PRECONDITION_COVERAGE": "present",
                },
                path=log,
            )
            self.assertIsNotNone(event_id)
            event = _load_control_event(log, str(event_id))
            append_review(
                event=event,
                action="OVERRIDE",
                human_level="MEDIUM",
                reason="Boundary conditions manually verified",
                path=log,
            )

            records = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["record_type"], "control_event")
            self.assertEqual(records[1]["record_type"], "human_review")
            self.assertEqual(records[0]["event_id"], records[1]["event_id"])


    def test_p002_uses_dpg_derived_numeric_predicates(self) -> None:
        decision = apply_control(
            score=0.538,
            original_level="MEDIUM",
            concept_states={
                "CONTRACT_COVERAGE": "present",
                "POSTCONDITION_COMPLEXITY": "present",
            },
            feature_values={
                "n_contracts_total": 2,
                "ensures_complexity": 10,
                "n_loc": 6,
            },
            policy_document=self.policies,
            low_threshold=0.20,
            high_threshold=0.60,
        )

        self.assertEqual(decision.model_score, 0.538)
        self.assertEqual(decision.controlled_level, "HIGH")
        self.assertTrue(decision.decision_changed)
        self.assertTrue(decision.review_required)
        self.assertIn("P002", decision.matched_policy_ids)
        self.assertNotIn("P003", decision.matched_policy_ids)

    def test_p003_uses_dpg_derived_numeric_predicates(self) -> None:
        decision = apply_control(
            score=0.547,
            original_level="MEDIUM",
            concept_states={
                "STRUCTURAL_SIZE": "present",
                "POSTCONDITION_COMPLEXITY": "present",
            },
            feature_values={
                "n_contracts_total": 5,
                "n_loc": 11,
                "ensures_complexity": 39,
            },
            policy_document=self.policies,
            low_threshold=0.20,
            high_threshold=0.60,
        )

        self.assertEqual(decision.model_score, 0.547)
        self.assertEqual(decision.controlled_level, "HIGH")
        self.assertTrue(decision.decision_changed)
        self.assertTrue(decision.review_required)
        self.assertIn("P003", decision.matched_policy_ids)

    def test_matching_policy_without_decision_change_does_not_require_review(self) -> None:
        decision = apply_control(
            score=0.30,
            original_level="MEDIUM",
            concept_states={
                "CONTRACT_COVERAGE": "present",
                "POSTCONDITION_COMPLEXITY": "present",
            },
            feature_values={
                "n_contracts_total": 2,
                "ensures_complexity": 14,
                "n_loc": 6,
            },
            policy_document=self.policies,
            low_threshold=0.20,
            high_threshold=0.60,
        )

        self.assertIn("P002", decision.matched_policy_ids)
        self.assertEqual(decision.controlled_high_threshold, 0.50)
        self.assertEqual(decision.controlled_level, "MEDIUM")
        self.assertFalse(decision.decision_changed)
        self.assertFalse(decision.review_required)


if __name__ == "__main__":
    unittest.main()
