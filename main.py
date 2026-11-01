"""Daily arXiv digest: fetch new papers, rank them with Gemini against your research, email the result."""

import argparse
import json
import os
import smtplib
from datetime import date, datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path

import yaml
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader

from arxiv import fetch_api, fetch_rss
from ranker import rank

ROOT = Path(__file__).parent
# arXiv ids already emailed. Every run skips these, so repeated or early runs never resend a paper.
STATE = ROOT / "state" / "sent.json"
RETENTION = timedelta(days=30)
DEFAULT_MODEL = "gemini-2.5-flash"


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT.parent / ".env")
    config = yaml.safe_load(Path(os.environ.get("DIGEST_CONFIG", ROOT.parent / "config.yaml")).read_text())
    categories = config.get("arxiv_categories", ["cs.DB"])

    if args.since:
        papers = fetch_api(categories, args.since)
        label = f"since {args.since:%Y-%m-%d}"
    else:
        papers, batch_date = fetch_rss(categories)
        label = str(batch_date)

    sent = json.loads(STATE.read_text()) if STATE.exists() else {}
    papers = [p for p in papers if p.id not in sent]
    print(f"[main] {label}: {len(papers)} unsent paper(s)")
    if not papers:
        return

    tiers = rank(papers, config["research_description"], config.get("llm_model", DEFAULT_MODEL))
    html = Environment(loader=FileSystemLoader(ROOT / "templates"), autoescape=True).get_template("digest.html").render(
        label=label, total=len(papers), categories=categories, **tiers,
    )
    subject = f"DB Paper Digest – {label} | {len(papers)} new paper{'s' if len(papers) != 1 else ''}"
    if args.dry_run:
        Path("preview.html").write_text(html)
        print(f"[main] Dry run: wrote preview.html ({subject!r})")
        return

    send(config["sender_email"], config["recipient_email"], subject, html)
    today = date.today()
    sent.update({p.id: today.isoformat() for p in papers})
    sent = {i: d for i, d in sent.items() if date.fromisoformat(d) > today - RETENTION}
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(sent, indent=0, sort_keys=True))


def send(sender: str, recipient: str, subject: str, html: str) -> None:
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, sender, recipient
    msg.set_content(html, subtype="html")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender, os.environ["GMAIL_APP_PASSWORD"])
        smtp.send_message(msg)
    print(f"[main] Sent {subject!r}")


def utc_datetime(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--since", type=utc_datetime, metavar="YYYY-MM-DD[THH:MM]",
                   help="backfill: all papers submitted since this time (UTC) instead of today's RSS batch")
    p.add_argument("--dry-run", action="store_true", help="write preview.html instead of emailing")
    return p.parse_args()


if __name__ == "__main__":
    main()
