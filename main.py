import argparse
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import yaml
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader

from pipeline.email_sender import send_digest_email, send_empty_email
from pipeline.ranker import rank_papers
from pipeline.state import State, load_state, save_state
from sources.arxiv import fetch_arxiv, fetch_arxiv_day

_REPO_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = Path(os.environ.get("DIGEST_CONFIG") or _REPO_ROOT / "config.yaml")
_STATE_PATH = Path(os.environ.get("DIGEST_STATE") or Path(__file__).parent / "state" / "seen.json")


def main() -> None:
    args = _parse_args()
    if (_env := _REPO_ROOT / ".env").exists():
        load_dotenv(_env)
    config = _load_config(_CONFIG_PATH)
    _validate_env()

    start_date, end_date = _compute_date_range(args)
    print(f"[main] Fetching papers for {start_date} – {end_date}")

    t0 = time.perf_counter()
    papers, warnings, batch_date = fetch_arxiv(config, start_date, end_date)
    print(f"[main] {len(papers)} papers fetched (batch_date={batch_date})")
    for w in warnings:
        print(f"[main] WARNING: {w}", file=sys.stderr)

    # Dry run: pure preview of the current fetch — no gate, no dedup, no state.
    if args.dry_run:
        _emit(config, papers, batch_date or end_date, warnings, args)
    # Manual --date/--since: the user asked for a specific range, so honour it
    # verbatim and stay out of the idempotency machinery.
    elif args.date or args.since:
        _emit(config, papers, end_date, warnings, args)
    else:
        _run_scheduled(config, papers, warnings, batch_date, end_date, args)

    print(f"[timing] total: {time.perf_counter() - t0:.2f}s")


def _run_scheduled(config: dict, papers, warnings, batch_date, end_date, args) -> None:
    """Default daily path, made idempotent per announcement batch.

    Correctness here comes from the ``batch_date`` watermark, not from timing:
    each distinct arXiv batch is emailed exactly once, whenever a run first sees
    it — so early/duplicate polls, delayed runs, and dropped runs can't cause
    duplicates or stale resends.
    """
    state = load_state(_STATE_PATH)
    today = date.today()
    watermark = state.last_batch_date

    # Stale early poll: the live feed still shows an older batch than we've
    # already processed (a run that fired before arXiv's rebuild). Exit silently
    # — never resend, never send an empty "nothing new" mail.
    if batch_date is not None and watermark is not None and batch_date < watermark:
        print(f"[main] Feed batch {batch_date} < watermark {watermark}; stale poll, skipping.")
        return

    is_new_batch = batch_date is None or watermark is None or batch_date > watermark

    # Recover any announcement days we missed between the watermark and now
    # (e.g. every poll was dropped one morning), each as its own dated digest.
    if batch_date is not None and watermark is not None:
        _backfill_gap(config, state, watermark, batch_date, args)

    # De-dup by seen-ID rather than trusting batch_date alone. This handles both
    # a repeat poll of the same batch (== watermark → all seen → nothing sent)
    # and a batch that filled in over successive rebuilds (same batch_date, new
    # papers appear → we send only the newcomers instead of skipping them).
    digest_date = batch_date or end_date
    fresh = [p for p in papers if not state.is_seen(p.id)]
    dropped = len(papers) - len(fresh)
    if dropped:
        print(f"[main] Dropped {dropped} already-seen paper(s) from batch {digest_date}")

    if fresh:
        _emit(config, fresh, digest_date, warnings, args)
        state.mark((p.id for p in fresh), digest_date)
    elif papers:
        print(f"[main] No new papers in batch {digest_date} (all already emailed) — skipping.")
    elif is_new_batch:
        send_empty_email(config, digest_date)

    if is_new_batch and batch_date is not None:
        state.last_batch_date = batch_date

    # Persist only when something actually changed, so idempotent no-op polls
    # don't churn the committed state file.
    if fresh or (is_new_batch and batch_date is not None):
        save_state(_STATE_PATH, state, today)
        print(f"[main] State saved: watermark={state.last_batch_date}, tracked={len(state.seen)}")
    else:
        print("[main] No change — state left as-is.")


def _backfill_gap(config: dict, state: State, watermark: date, batch_date: date, args) -> None:
    day = watermark + timedelta(days=1)
    while day < batch_date:
        papers, warnings = fetch_arxiv_day(config, day)
        fresh = [p for p in papers if not state.is_seen(p.id)]
        if fresh:
            print(f"[main] Backfilling {len(fresh)} missed paper(s) for {day}")
            _emit(config, fresh, day, warnings, args)
            state.mark((p.id for p in fresh), day)
        day += timedelta(days=1)


def _emit(config: dict, papers, digest_date: date, warnings, args) -> None:
    if not papers:
        if args.dry_run:
            print("[main] No papers — dry run, skipping empty digest email.")
        else:
            send_empty_email(config, digest_date)
        return
    ranked = rank_papers(papers, config)
    html = _render_digest(config, ranked, digest_date, len(papers), warnings)
    if args.dry_run:
        print("\n" + "=" * 72 + "\n" + html + "\n" + "=" * 72)
        print("[main] Dry run — email not sent.")
    else:
        send_digest_email(config, html, digest_date, len(papers))


def _compute_date_range(args: argparse.Namespace) -> tuple[date, date]:
    if args.date and args.since:
        sys.exit("[main] --date and --since are mutually exclusive.")
    if args.date:
        d = date.fromisoformat(args.date)
        return d, d
    if args.since:
        start = date.fromisoformat(args.since)
        end = date.today() - timedelta(days=1)
        if start > end:
            sys.exit(f"[main] --since date {start} is in the future.")
        return start, end
    today = date.today()
    # Monday's arxiv batch (Fri–Mon submissions) is dated today, not yesterday.
    d = today if today.weekday() == 0 else today - timedelta(days=1)
    return d, d


def _render_digest(config: dict, ranked, digest_date: date, total: int, warnings: list[str] | None = None) -> str:
    from datetime import datetime
    template_file = config.get("template_file", "templates/digest.html")
    env = Environment(loader=FileSystemLoader(os.path.dirname(template_file)), autoescape=True)
    return env.get_template(os.path.basename(template_file)).render(
        digest_date=digest_date,
        total=total,
        must_read=ranked.must_read,
        skim=ranked.skim,
        irrelevant=ranked.irrelevant,
        warnings=warnings or [],
        generation_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


def _load_config(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _validate_env() -> None:
    missing = [k for k in ("GEMINI_API_KEY", "GMAIL_APP_PASSWORD") if not os.environ.get(k)]
    if missing:
        sys.exit(f"[main] Missing env vars: {', '.join(missing)}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--date", metavar="YYYY-MM-DD", default=None)
    p.add_argument("--since", metavar="YYYY-MM-DD", default=None)
    return p.parse_args()


if __name__ == "__main__":
    main()
