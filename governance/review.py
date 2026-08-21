"""Human review CLI for append-only SpecLens-PML governance events."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_PATH = ROOT / "data" / "governance" / "control_events.jsonl"
VALID_LEVELS = ("LOW", "MEDIUM", "HIGH")


def _load_control_event(path: Path, event_id: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Governance log not found: {path}")

    found: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record_type") == "control_event" and record.get("event_id") == event_id:
                found = record

    if found is None:
        raise ValueError(f"Control event not found: {event_id}")
    return found


def append_review(
    *,
    event: dict[str, Any],
    action: str,
    human_level: str,
    reason: str,
    path: Path,
) -> dict[str, Any]:
    record = {
        "record_type": "human_review",
        "review_id": f"REV-{uuid4().hex[:12]}",
        "event_id": event["event_id"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "review_action": action,
        "controlled_level": event["controlled_level"],
        "human_level": human_level,
        "reason": reason.strip(),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review a SpecLens-PML control event.")
    parser.add_argument("event_id", help="Control event identifier (EVT-...).")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--approve", action="store_true", help="Approve the controlled decision.")
    action.add_argument("--override", choices=VALID_LEVELS, help="Override the controlled decision.")
    parser.add_argument("--reason", default="", help="Human rationale; required for overrides.")
    parser.add_argument("--log", type=Path, default=DEFAULT_AUDIT_PATH, help="Governance JSONL log path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    event = _load_control_event(args.log, args.event_id)

    if args.override and not args.reason.strip():
        raise SystemExit("--reason is required when using --override.")

    if args.approve:
        action = "APPROVE"
        human_level = str(event["controlled_level"])
        reason = args.reason or "Controlled decision approved by human reviewer."
    else:
        action = "OVERRIDE"
        human_level = str(args.override)
        reason = args.reason

    review = append_review(
        event=event,
        action=action,
        human_level=human_level,
        reason=reason,
        path=args.log,
    )

    print(f"Review recorded: {review['review_id']}")
    print(f"Control event: {review['event_id']}")
    print(f"Action: {review['review_action']}")
    print(f"Controlled level: {review['controlled_level']}")
    print(f"Final human level: {review['human_level']}")
    print(f"Reason: {review['reason']}")


if __name__ == "__main__":
    main()
