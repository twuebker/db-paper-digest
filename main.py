"""db-paper-digest: daily database research paper digest via email."""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import yaml
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader

_REPO_ROOT = Path(__file__).parent.parent

# Config path: explicit env var (CI) → parent-dir default (local)
_CONFIG_PATH = Path(os.environ.get("DIGEST_CONFIG") or _REPO_ROOT / "config.yaml")

from pipeline.dedup import dedup_same_day
from pipeline.email_sender import send_digest_email, send_empty_email
from pipeline.ranker import rank_papers
from sources.arxiv import fetch_arxiv
from sources.crossref import fetch_crossref


def main() -> None:
    args = _parse_args()
    # .env is only needed locally; in CI secrets arrive as real env vars
    _env_file = _REPO_ROOT / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
    config = _load_config(_CONFIG_PATH)
    _validate_env()

    start_date, end_date = _compute_date_range(args)
    print(f"[main] Fetching papers for {start_date} – {end_date}")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(fetch_arxiv, config, start_date, end_date): "arxiv",
            executor.submit(fetch_crossref, config, start_date, end_date): "crossref",
        }
        arxiv_papers, crossref_papers = [], []
        for future in as_completed(futures):
            source = futures[future]
            try:
                result = future.result()
                print(f"[main] {source}: {len(result)} papers fetched")
                if source == "arxiv":
                    arxiv_papers = result
                else:
                    crossref_papers = result
            except Exception as exc:
                print(f"[main] WARNING: {source} fetch failed: {exc}", file=sys.stderr)

    papers = dedup_same_day(arxiv_papers + crossref_papers)
    print(f"[main] {len(papers)} papers after deduplication")

    if not papers:
        if args.dry_run:
            print("[main] No new papers — dry run, skipping empty digest email.")
        else:
            print("[main] No new papers — sending empty digest.")
            send_empty_email(config, end_date)
        return

    ranked = rank_papers(papers, config)
    html = _render_digest(config, ranked, end_date, len(papers))

    if args.dry_run:
        print("\n" + "=" * 72)
        print(html)
        print("=" * 72)
        print("[main] Dry run — email not sent.")
    else:
        send_digest_email(config, html, end_date, len(papers))


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
    if today.weekday() == 0:  # Monday — cover the weekend
        start = today - timedelta(days=3)  # Friday
        end = today - timedelta(days=1)    # Sunday
    else:
        yesterday = today - timedelta(days=1)
        start = end = yesterday
    return start, end


def _render_digest(config: dict, ranked, digest_date: date, total: int) -> str:
    from datetime import datetime
    template_file = config.get("template_file", "templates/digest.html")
    template_dir = os.path.dirname(template_file)
    template_name = os.path.basename(template_file)

    env = Environment(loader=FileSystemLoader(template_dir), autoescape=True)
    template = env.get_template(template_name)
    return template.render(
        digest_date=digest_date,
        total=total,
        must_read=ranked.must_read,
        skim=ranked.skim,
        irrelevant=ranked.irrelevant,
        generation_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _validate_env() -> None:
    missing = [k for k in ("CSCS_SERVING_API", "GMAIL_APP_PASSWORD") if not os.environ.get(k)]
    if missing:
        sys.exit(f"[main] Missing required environment variables: {', '.join(missing)}\n"
                 f"       Set them in your .env file.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a daily DB paper digest email.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the rendered HTML to stdout instead of sending email.",
    )
    parser.add_argument(
        "--date",
        metavar="YYYY-MM-DD",
        default=None,
        help="Fetch papers for a specific date instead of yesterday.",
    )
    parser.add_argument(
        "--since",
        metavar="YYYY-MM-DD",
        default=None,
        help="Fetch papers from this date through yesterday (catch-up mode).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
