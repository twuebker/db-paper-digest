"""Fetch ACM conference papers from the Crossref API."""

import html
import re
import time as time_mod
from datetime import date

import requests

from sources import Paper

CROSSREF_BASE = "https://api.crossref.org/works"
ROWS_PER_PAGE = 100
MAX_RETRIES = 3


def fetch_crossref(config: dict, start_date: date, end_date: date) -> list[Paper]:
    conferences = config.get("crossref_conferences", ["SIGMOD", "VLDB", "ICDE", "EDBT", "CIDR"])
    sender_email = config.get("sender_email", "")
    date_str_start = start_date.strftime("%Y-%m-%d")
    date_str_end = end_date.strftime("%Y-%m-%d")

    papers: list[Paper] = []
    cursor = "*"

    while True:
        params = {
            "filter": f"from-pub-date:{date_str_start},until-pub-date:{date_str_end}",
            "select": "DOI,title,abstract,author,container-title,URL,published",
            "rows": ROWS_PER_PAGE,
            "cursor": cursor,
        }
        if sender_email:
            params["mailto"] = sender_email

        t0 = time_mod.perf_counter()
        data = _fetch_page(params)
        message = data.get("message", {})
        items = message.get("items", [])
        print(f"[timing] crossref page (cursor={cursor[:8]}…): {time_mod.perf_counter() - t0:.2f}s, {len(items)} items")

        for item in items:
            paper = _parse_item(item)
            if paper and _matches_conference(item, conferences):
                papers.append(paper)

        next_cursor = message.get("next-cursor")
        if not items or not next_cursor:
            break
        cursor = next_cursor
        time_mod.sleep(1)  # be polite to Crossref

    return papers


def _fetch_page(params: dict) -> dict:
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(CROSSREF_BASE, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except (requests.HTTPError, requests.JSONDecodeError) as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(f"[crossref] Error {exc}, retrying in {wait}s…")
            time.sleep(wait)
    return {}


def _parse_item(item: dict) -> Paper | None:
    titles = item.get("title", [])
    if not titles:
        return None
    title = titles[0].strip()
    if not title:
        return None

    doi = (item.get("DOI") or "").lower().strip()
    url = item.get("URL") or (f"https://doi.org/{doi}" if doi else "")

    raw_abstract = item.get("abstract", "")
    abstract = _strip_jats(raw_abstract) if raw_abstract else None

    authors = []
    for a in item.get("author", []):
        given = a.get("given", "")
        family = a.get("family", "")
        name = f"{given} {family}".strip()
        if name:
            authors.append(name)

    container_titles = item.get("container-title", [])
    venue = container_titles[0] if container_titles else None

    if not doi:
        return None

    return Paper(
        id=doi,
        title=title,
        abstract=abstract,
        authors=authors,
        url=url,
        source="crossref",
        venue=venue,
    )


def _matches_conference(item: dict, conferences: list[str]) -> bool:
    container_titles = " ".join(item.get("container-title", [])).lower()
    return any(c.lower() in container_titles for c in conferences)


def _strip_jats(text: str) -> str:
    """Remove JATS XML tags and unescape HTML entities."""
    stripped = re.sub(r"<[^>]+>", "", text)
    return html.unescape(stripped).strip()
