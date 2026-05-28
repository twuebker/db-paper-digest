import html
import re
import time as time_mod
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import requests

from sources import Paper

ARXIV_BASE = "https://export.arxiv.org/api/query"
RSS_BASE = "https://rss.arxiv.org/rss"
ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
DC_NS = "{http://purl.org/dc/elements/1.1/}"
BATCH_SIZE = 100
MAX_RETRIES = 5
USER_AGENT = "db-paper-digest/1.0 (github.com/twubker/db-paper-digest)"

_VENUE_PATTERNS = [
    (r"\bSIGMOD\b", "SIGMOD"),
    (r"\bVLDB\b|\bPVLDB\b", "VLDB"),
    (r"\bICDE\b", "ICDE"),
    (r"\bEDBT\b", "EDBT"),
    (r"\bCIDR\b", "CIDR"),
]
_ARXIV_ID_RE = re.compile(r'arXiv:(\d{4}\.\d+)', re.IGNORECASE)
_STRIP_HTML = re.compile(r'<[^>]+>')


def fetch_arxiv(config: dict, start_date: date, end_date: date) -> tuple[list[Paper], list[str]]:
    categories = config.get("arxiv_categories", ["cs.DB", "cs.IR"])
    papers, warnings, batch_date = _try_rss(categories)
    if batch_date is not None and batch_date >= start_date - timedelta(days=1):
        print(f"[arxiv] RSS: batch_date={batch_date}, {len(papers)} papers")
        return papers, warnings
    print(f"[arxiv] RSS miss (batch_date={batch_date}) — using export API")
    return _fetch_export_api(categories, start_date, end_date)


def _try_rss(categories: list[str]) -> tuple[list[Paper], list[str], date | None]:
    seen: dict[str, Paper] = {}
    warnings: list[str] = []
    batch_date: date | None = None
    for category in categories:
        try:
            resp = requests.get(f"{RSS_BASE}/{category}", timeout=30, headers={"User-Agent": USER_AGENT})
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            warnings.append(f"RSS failed for {category}: {exc}")
            continue
        cat_papers, cat_warnings, cat_date = _parse_rss(resp.content, category)
        warnings.extend(cat_warnings)
        if cat_date is not None:
            batch_date = cat_date
        for p in cat_papers:
            seen.setdefault(p.id, p)
    return list(seen.values()), warnings, batch_date


def _parse_rss(xml_bytes: bytes, category: str) -> tuple[list[Paper], list[str], date | None]:
    root = ET.fromstring(xml_bytes)
    channel = root.find("channel")
    if channel is None:
        return [], [f"RSS: no <channel> for {category}"], None

    batch_date: date | None = None
    for tag in ("lastBuildDate", "pubDate"):
        el = channel.find(tag)
        if el is not None and el.text:
            try:
                batch_date = parsedate_to_datetime(el.text.strip()).astimezone(timezone.utc).date()
                break
            except Exception:
                pass

    papers, warnings = [], []
    for item in channel.findall("item"):
        title_raw = (item.findtext("title") or "").strip()
        m = _ARXIV_ID_RE.search(title_raw)
        if not m:
            warnings.append(f"RSS: no arXiv ID in title: {title_raw!r}")
            continue
        arxiv_id = m.group(1)
        title = re.sub(r'\s*\(arXiv:[^\)]+\)\s*$', '', title_raw).strip() or title_raw
        description = (item.findtext("description") or "").strip()
        abstract = html.unescape(_STRIP_HTML.sub("", description)).strip() or None
        creator = (item.findtext(f"{DC_NS}creator") or "").strip()
        authors = [a.strip() for a in creator.split(",") if a.strip()]
        link = (item.findtext("link") or "").strip()
        url = re.sub(r'v\d+$', '', link.replace("http://", "https://")) if link else f"https://arxiv.org/abs/{arxiv_id}"
        comment_el = item.find(f"{ARXIV_NS}comment")
        comment = " ".join((comment_el.text or "").split()) if comment_el is not None else None
        papers.append(Paper(
            id=arxiv_id, title=title, abstract=abstract, authors=authors,
            url=url, source="arxiv", venue=_detect_venue(comment, None), comment=comment,
        ))
    return papers, warnings, batch_date


def _fetch_export_api(categories: list[str], start_date: date, end_date: date) -> tuple[list[Paper], list[str]]:
    cat_query = " OR ".join(f"cat:{c}" for c in categories)
    qs = (start_date - timedelta(days=1)).strftime("%Y%m%d") + "000000"
    qe = (end_date + timedelta(days=1)).strftime("%Y%m%d") + "235959"
    search_query = f"({cat_query}) AND submittedDate:[{qs} TO {qe}]"

    raw, warnings, offset = [], [], 0
    while True:
        batch, batch_warnings = _fetch_batch({
            "search_query": search_query, "start": offset,
            "max_results": BATCH_SIZE, "sortBy": "submittedDate", "sortOrder": "descending",
        })
        raw.extend(batch)
        warnings.extend(batch_warnings)
        if len(batch) < BATCH_SIZE:
            break
        offset += BATCH_SIZE
        time_mod.sleep(3)

    papers = [p for p, announced in raw if start_date <= announced <= end_date]
    print(f"[arxiv] export API: {len(raw)} fetched, {len(papers)} in window")
    return papers, warnings


def _fetch_batch(params: dict) -> tuple[list[tuple[Paper, date]], list[str]]:
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(ARXIV_BASE, params=params, timeout=60, headers={"User-Agent": USER_AGENT})
            resp.raise_for_status()
            return _parse_atom(resp.content)
        except requests.exceptions.HTTPError as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            if exc.response is not None and exc.response.status_code == 429:
                ra = exc.response.headers.get("Retry-After")
                wait = int(ra) if ra and ra.isdigit() else 60 * (attempt + 1)
                print(f"[arxiv] 429 — waiting {wait}s (retry {attempt + 2}/{MAX_RETRIES})")
            else:
                wait = 2 ** (attempt + 2)
                print(f"[arxiv] HTTP error (attempt {attempt + 1}/{MAX_RETRIES}): {exc} — retrying in {wait}s")
            time_mod.sleep(wait)
        except requests.exceptions.RequestException as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** (attempt + 2)
            print(f"[arxiv] request failed (attempt {attempt + 1}/{MAX_RETRIES}): {exc} — retrying in {wait}s")
            time_mod.sleep(wait)
    return [], []


def _parse_atom(xml_bytes: bytes) -> tuple[list[tuple[Paper, date]], list[str]]:
    root = ET.fromstring(xml_bytes)
    papers, warnings = [], []
    for entry in root.findall(f"{ATOM_NS}entry"):
        raw_id = (entry.findtext(f"{ATOM_NS}id") or "").strip()
        arxiv_id = re.sub(r'https?://arxiv\.org/abs/', '', raw_id).split("v")[0]
        title_el = entry.find(f"{ATOM_NS}title")
        title = " ".join((title_el.text or "").split()) if title_el is not None else ""
        abstract_el = entry.find(f"{ATOM_NS}summary")
        abstract = " ".join((abstract_el.text or "").split()) if abstract_el is not None else None
        authors = [
            (name_el.text or "").strip()
            for a in entry.findall(f"{ATOM_NS}author")
            if (name_el := a.find(f"{ATOM_NS}name")) is not None
        ]
        url = next(
            (lnk.get("href", "") for lnk in entry.findall(f"{ATOM_NS}link") if lnk.get("rel") == "alternate"),
            f"https://arxiv.org/abs/{arxiv_id}",
        )
        published_el = entry.find(f"{ATOM_NS}published")
        published_str = (published_el.text or "").strip() if published_el is not None else ""
        try:
            announced = datetime.fromisoformat(published_str.replace("Z", "+00:00")).astimezone(timezone.utc).date()
        except ValueError:
            warnings.append(f"Skipped — bad date ({published_str!r}): {title or arxiv_id}")
            continue
        comment_el = entry.find(f"{ARXIV_NS}comment")
        comment = " ".join((comment_el.text or "").split()) if comment_el is not None else None
        jref_el = entry.find(f"{ARXIV_NS}journal_ref")
        journal_ref = " ".join((jref_el.text or "").split()) if jref_el is not None else None
        if not arxiv_id or not title:
            warnings.append(f"Skipped — missing ID or title (id={arxiv_id!r})")
            continue
        papers.append((Paper(
            id=arxiv_id, title=title, abstract=abstract, authors=authors, url=url,
            source="arxiv", venue=_detect_venue(comment, journal_ref), comment=comment, journal_ref=journal_ref,
        ), announced))
    return papers, warnings


def _detect_venue(comment: str | None, journal_ref: str | None) -> str | None:
    haystack = " ".join(filter(None, [comment, journal_ref])).upper()
    if not haystack:
        return None
    for pattern, short_name in _VENUE_PATTERNS:
        if re.search(pattern, haystack):
            year = re.search(r"\b(20\d{2})\b", haystack)
            return f"Accepted at {short_name}" + (f" {year.group(1)}" if year else "")
    return None
