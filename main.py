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
from sources.arxiv import fetch_arxiv

_REPO_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = Path(os.environ.get("DIGEST_CONFIG") or _REPO_ROOT / "config.yaml")


def main() -> None:
    args = _parse_args()
    if (_env := _REPO_ROOT / ".env").exists():
        load_dotenv(_env)
    config = _load_config(_CONFIG_PATH)
    _validate_env()

    start_date, end_date = _compute_date_range(args)
    print(f"[main] Fetching papers for {start_date} – {end_date}")

    t0 = time.perf_counter()
    papers, warnings = fetch_arxiv(config, start_date, end_date)
    print(f"[main] {len(papers)} papers fetched")
    for w in warnings:
        print(f"[main] WARNING: {w}", file=sys.stderr)

    if not papers:
        if args.dry_run:
            print("[main] No new papers — dry run, skipping empty digest email.")
        else:
            send_empty_email(config, end_date)
        return

    ranked = rank_papers(papers, config)
    html = _render_digest(config, ranked, end_date, len(papers), warnings)

    if args.dry_run:
        print("\n" + "=" * 72 + "\n" + html + "\n" + "=" * 72)
        print("[main] Dry run — email not sent.")
    else:
        send_digest_email(config, html, end_date, len(papers))

    print(f"[timing] total: {time.perf_counter() - t0:.2f}s")


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
    if today.weekday() == 0:  # Monday: cover the weekend
        return today - timedelta(days=3), today - timedelta(days=1)
    yesterday = today - timedelta(days=1)
    return yesterday, yesterday


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
