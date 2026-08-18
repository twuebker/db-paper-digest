"""Persistent cross-run state, kept in the GitHub Actions cache between runs.

GitHub Actions runners are ephemeral, so we can't remember across runs what we
already emailed unless we persist it somewhere durable. The workflow restores
and saves this tiny JSON file via actions/cache (no repo commits). It holds two
things:

* ``last_batch_date`` — the arXiv RSS ``batch_date`` of the most recent batch we
  processed. This is the watermark that makes each daily batch get emailed
  *exactly once*: a run only acts when the live feed shows a newer batch_date.
* ``seen`` — a rolling map of ``arxiv_id -> announcement date`` for the last
  ``RETENTION_DAYS`` days, used as a fine-grained de-dup safety net (e.g. when a
  transitional feed read overlaps the previous day) and to skip already-emailed
  papers during gap backfill.
"""

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

RETENTION_DAYS = 10


@dataclass
class State:
    last_batch_date: date | None = None
    seen: dict[str, str] = field(default_factory=dict)  # arxiv_id -> ISO announcement date

    def is_seen(self, arxiv_id: str) -> bool:
        return arxiv_id in self.seen

    def mark(self, arxiv_ids, announced: date) -> None:
        for aid in arxiv_ids:
            self.seen[aid] = announced.isoformat()


def load_state(path: Path) -> State:
    if not path.exists():
        return State()
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, ValueError):
        return State()
    return State(
        last_batch_date=_safe_date(data.get("last_batch_date")),
        seen=dict(data.get("seen", {})),
    )


def save_state(path: Path, state: State, today: date) -> None:
    cutoff = today - timedelta(days=RETENTION_DAYS)
    pruned = {
        aid: d for aid, d in state.seen.items()
        if (parsed := _safe_date(d)) is None or parsed >= cutoff
    }
    state.seen = pruned
    payload = {
        "last_batch_date": state.last_batch_date.isoformat() if state.last_batch_date else None,
        "seen": dict(sorted(pruned.items())),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _safe_date(value) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except (ValueError, TypeError):
        return None
