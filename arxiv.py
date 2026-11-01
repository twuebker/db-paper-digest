"""Fetch papers from arXiv: the latest daily batch via RSS, or a backfill window via the export API."""

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime

import requests

RSS_URL = "https://rss.arxiv.org/rss/{}"
API_URL = "https://export.arxiv.org/api/query"
HEADERS = {"User-Agent": "db-paper-digest (https://github.com/twuebker/db-paper-digest)"}
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"
DC = "{http://purl.org/dc/elements/1.1/}"
PAGE_SIZE = 200


@dataclass
class Paper:
    id: str
    title: str
    abstract: str
    authors: list[str]

    @property
    def url(self) -> str:
        return f"https://arxiv.org/abs/{self.id}"


def fetch_rss(categories: list[str]) -> tuple[list[Paper], date | None]:
    """Return the feed's current announcement batch (new submissions and cross-lists) and its date.

    The feed only ever holds the latest batch and is empty on days without an announcement.
    Replacements of older papers are skipped: they were already announced when first submitted.
    """
    papers: dict[str, Paper] = {}
    batch_date = None
    for category in categories:
        channel = ET.fromstring(_get(RSS_URL.format(category))).find("channel")
        batch_date = parsedate_to_datetime(channel.findtext("pubDate")).date()
        for item in channel.iter("item"):
            if item.findtext(f"{ARXIV}announce_type") not in ("new", "cross"):
                continue
            arxiv_id = item.findtext("link").rstrip("/").split("/")[-1]
            abstract = item.findtext("description").split("Abstract:", 1)[-1]
            papers.setdefault(arxiv_id, Paper(
                id=arxiv_id,
                title=_clean(item.findtext("title")),
                abstract=_clean(abstract),
                authors=[a.strip() for a in item.findtext(f"{DC}creator", "").split(",") if a.strip()],
            ))
    return list(papers.values()), batch_date


def fetch_api(categories: list[str], since: datetime) -> list[Paper]:
    """Return all papers first submitted since `since` (UTC) that arXiv has announced so far."""
    cats = " OR ".join(f"cat:{c}" for c in categories)
    query = f"({cats}) AND submittedDate:[{since:%Y%m%d%H%M} TO {datetime.now(timezone.utc):%Y%m%d%H%M}]"
    papers: dict[str, Paper] = {}
    for start in range(0, 10_000, PAGE_SIZE):
        feed = ET.fromstring(_get(API_URL, {
            "search_query": query, "start": start, "max_results": PAGE_SIZE,
            "sortBy": "submittedDate", "sortOrder": "ascending",
        }))
        entries = feed.findall(f"{ATOM}entry")
        for e in entries:
            arxiv_id = re.sub(r"v\d+$", "", e.findtext(f"{ATOM}id").split("/abs/")[-1])
            papers.setdefault(arxiv_id, Paper(
                id=arxiv_id,
                title=_clean(e.findtext(f"{ATOM}title")),
                abstract=_clean(e.findtext(f"{ATOM}summary")),
                authors=[a.findtext(f"{ATOM}name").strip() for a in e.findall(f"{ATOM}author")],
            ))
        if len(entries) < PAGE_SIZE:
            break
        time.sleep(3)  # arXiv API etiquette: at most one request every 3 seconds
    return list(papers.values())


def _get(url: str, params: dict | None = None, attempts: int = 5) -> bytes:
    for attempt in range(attempts):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as exc:
            if attempt == attempts - 1:
                raise
            wait = 15 * 2 ** attempt
            print(f"[arxiv] {exc} — retrying in {wait}s")
            time.sleep(wait)


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())
