"""Fetch yesterday's preprints from the arXiv Atom feed API.

The arXiv Atom feed includes two extra fields beyond title/abstract:
  <arxiv:comment>   — free-text note from the author, often contains
                      "Accepted at SIGMOD 2026" or "To appear in VLDB 2026"
  <arxiv:journal_ref> — structured venue reference once published

We parse both, run a venue-detection heuristic, and set paper.venue to a
short label (e.g. "Accepted at SIGMOD 2026") when found.  This turns the
cs.DB feed into a near-real-time "just accepted" signal for conference papers.
"""

import re
import time as time_mod
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone

import requests

from sources import Paper

ARXIV_BASE = "http://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
BATCH_SIZE = 100
MAX_RETRIES = 3

# Canonical short names we want to surface; order matters for display.
_VENUE_PATTERNS: list[tuple[str, str]] = [
    (r"\bSIGMOD\b", "SIGMOD"),
    (r"\bVLDB\b|\bPVLDB\b", "VLDB"),
    (r"\bICDE\b", "ICDE"),
    (r"\bEDBT\b", "EDBT"),
    (r"\bCIDR\b", "CIDR"),
]


def fetch_arxiv(config: dict, start_date: date, end_date: date) -> list[Paper]:
    categories = config.get("arxiv_categories", ["cs.DB", "cs.IR"])
    cat_query = " OR ".join(f"cat:{c}" for c in categories)
    # Query a 2-day window around the target range so we don't miss papers
    # whose submittedDate lags their announcement by a day (arXiv 14:00 ET
    # cutoff). We then filter by the `published` field (= actual announcement
    # date) after fetching, which is the authoritative "appeared in listing"
    # date and has no cross-day overlap between consecutive runs.
    query_start = start_date - timedelta(days=1)
    query_end = end_date + timedelta(days=1)
    start_str = query_start.strftime("%Y%m%d") + "000000"
    end_str = query_end.strftime("%Y%m%d") + "235959"
    search_query = f"({cat_query}) AND submittedDate:[{start_str} TO {end_str}]"

    raw: list[tuple[Paper, date]] = []
    offset = 0

    while True:
        params = {
            "search_query": search_query,
            "start": offset,
            "max_results": BATCH_SIZE,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        t0 = time_mod.perf_counter()
        batch = _fetch_batch(params)
        print(f"[timing] arxiv page (offset={offset}): {time_mod.perf_counter() - t0:.2f}s, {len(batch)} results")
        raw.extend(batch)
        if len(batch) < BATCH_SIZE:
            break
        offset += BATCH_SIZE
        time_mod.sleep(3)  # arXiv rate limit

    # Filter to papers actually announced within the target date range.
    papers = [
        paper for paper, announced in raw
        if start_date <= announced <= end_date
    ]
    print(f"[arxiv] {len(raw)} fetched, {len(papers)} within announcement window {start_date}–{end_date}")
    return papers


def _fetch_batch(params: dict) -> list[tuple[Paper, date]]:
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(ARXIV_BASE, params=params, timeout=30)
            resp.raise_for_status()
            return _parse_atom(resp.content)
        except requests.HTTPError as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(f"[arxiv] HTTP error {exc}, retrying in {wait}s…")
            time_mod.sleep(wait)
    return []


def _parse_atom(xml_bytes: bytes) -> list[tuple[Paper, date]]:
    root = ET.fromstring(xml_bytes)
    papers = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        raw_id = (entry.findtext(f"{ATOM_NS}id") or "").strip()
        # Strip URL prefix and version: "http://arxiv.org/abs/2401.12345v2" → "2401.12345"
        arxiv_id = raw_id.replace("http://arxiv.org/abs/", "").split("v")[0]

        title_el = entry.find(f"{ATOM_NS}title")
        title = " ".join((title_el.text or "").split()) if title_el is not None else ""

        abstract_el = entry.find(f"{ATOM_NS}summary")
        abstract = " ".join((abstract_el.text or "").split()) if abstract_el is not None else None

        authors = [
            (name_el.text or "").strip()
            for author_el in entry.findall(f"{ATOM_NS}author")
            if (name_el := author_el.find(f"{ATOM_NS}name")) is not None
        ]

        url = ""
        for link in entry.findall(f"{ATOM_NS}link"):
            if link.get("rel") == "alternate":
                url = link.get("href", "")
                break
        if not url:
            url = f"https://arxiv.org/abs/{arxiv_id}"

        # `published` = arXiv announcement date (authoritative listing date).
        # Format: "2024-01-15T00:00:00Z"
        published_el = entry.find(f"{ATOM_NS}published")
        published_str = (published_el.text or "").strip() if published_el is not None else ""
        try:
            announced = datetime.fromisoformat(published_str.replace("Z", "+00:00")).astimezone(timezone.utc).date()
        except ValueError:
            announced = None

        comment_el = entry.find(f"{ARXIV_NS}comment")
        comment = " ".join((comment_el.text or "").split()) if comment_el is not None else None

        jref_el = entry.find(f"{ARXIV_NS}journal_ref")
        journal_ref = " ".join((jref_el.text or "").split()) if jref_el is not None else None

        venue = _detect_venue(comment, journal_ref)

        if not arxiv_id or not title or announced is None:
            continue

        papers.append((Paper(
            id=arxiv_id,
            title=title,
            abstract=abstract if abstract else None,
            authors=authors,
            url=url,
            source="arxiv",
            venue=venue,
            comment=comment,
            journal_ref=journal_ref,
        ), announced))

    return papers


def _detect_venue(comment: str | None, journal_ref: str | None) -> str | None:
    """Return a short venue label if either field mentions a known conference."""
    haystack = " ".join(filter(None, [comment, journal_ref])).upper()
    if not haystack:
        return None

    for pattern, short_name in _VENUE_PATTERNS:
        if re.search(pattern, haystack):
            # Try to pull a year out of the surrounding text
            year_match = re.search(r"\b(20\d{2})\b", haystack)
            year = f" {year_match.group(1)}" if year_match else ""
            return f"Accepted at {short_name}{year}"

    return None
