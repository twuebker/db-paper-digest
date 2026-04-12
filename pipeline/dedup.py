"""Same-day cross-source deduplication.

When the same paper appears in both arXiv and Crossref on the same day
(e.g. a just-published conference paper that also has a preprint),
keep the Crossref version since it carries venue and DOI information.
"""

import re

from sources import Paper


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


def dedup_same_day(papers: list[Paper]) -> list[Paper]:
    """Deduplicate papers by normalized title.

    Crossref entries take priority over arXiv entries for the same title.
    Within the same source, first occurrence wins.
    """
    # Separate by source so crossref wins ties
    crossref = [p for p in papers if p.source == "crossref"]
    arxiv = [p for p in papers if p.source == "arxiv"]

    seen_titles: set[str] = set()
    result: list[Paper] = []

    for paper in crossref + arxiv:
        key = _normalize_title(paper.title)
        if not key:
            continue
        if key in seen_titles:
            continue
        seen_titles.add(key)
        result.append(paper)

    return result
