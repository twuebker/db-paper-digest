from dataclasses import dataclass


@dataclass
class Paper:
    id: str           # arXiv ID (e.g. "2401.12345") or DOI in lowercase
    title: str
    abstract: str | None
    authors: list[str]
    url: str
    source: str       # "arxiv" | "crossref"
    venue: str | None = None      # conference/journal name or "Accepted at X"
    comment: str | None = None    # raw arXiv comment field
    journal_ref: str | None = None  # raw arXiv journal-ref field


@dataclass
class RankedResult:
    must_read: list[dict]   # [{"paper": Paper, "summary": str}]
    skim: list[dict]        # max 10 items, same structure
    irrelevant: list[dict]  # [{"paper": Paper, "synopsis": str}]
