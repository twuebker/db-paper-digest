from dataclasses import dataclass


@dataclass
class Paper:
    id: str           # arXiv ID (e.g. "2401.12345") or DOI in lowercase
    title: str
    abstract: str | None
    authors: list[str]
    url: str
    source: str
    venue: str | None = None
    comment: str | None = None
    journal_ref: str | None = None
    is_replacement: bool = False


@dataclass
class RankedResult:
    must_read: list[dict]   # [{"paper": Paper, "summary": str}]
    skim: list[dict]        # max 10 items, same structure
    irrelevant: list[dict]  # [{"paper": Paper, "synopsis": str}]
