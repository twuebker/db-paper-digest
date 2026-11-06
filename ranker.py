"""Classify papers against the research description and summarize them with Gemini."""

import json
import os
import time

from google import genai
from google.genai import types
from pydantic import BaseModel

from arxiv import Paper

BATCH_SIZE = 100
MAX_SKIM = 10
ATTEMPTS = 4

PROMPT = """\
You are a research assistant helping a database researcher prioritize their daily reading.

The researcher's focus:
{research}

You will receive a JSON array of papers. Classify each paper into exactly one category:
- must_read: the single paper most relevant to the researcher's focus. Always pick exactly one; \
if nothing is relevant, pick the closest.
- skim: tangentially related or worth a quick look (at most {max_skim}).
- irrelevant: everything else.

Every input id must appear in exactly one category. For each paper write a "summary":
- must_read and skim: 2-3 sentences for a researcher who has 10 seconds to decide whether to click \
through. Lead with the specific method, system or finding (never "This paper proposes..."), state \
what was concretely built or shown and the key result, and, if space allows, what distinguishes it \
from prior work. Do not comment on relevance to the researcher.
- irrelevant: one sentence describing what the paper is about."""


class Entry(BaseModel):
    id: str
    summary: str


class Ranking(BaseModel):
    must_read: list[Entry]
    skim: list[Entry]
    irrelevant: list[Entry]


def rank(papers: list[Paper], research: str, model: str) -> dict[str, list[tuple[Paper, str]]]:
    """Return {"must_read" | "skim" | "irrelevant": [(paper, summary), ...]}.

    Large inputs (backfills) are ranked in batches of BATCH_SIZE, each contributing its own
    must-read and skim picks, so bigger digests surface proportionally more papers.
    """
    # system_instruction and response_schema need v1beta; the SDK's default version can change.
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=types.HttpOptions(api_version="v1beta"))
    config = types.GenerateContentConfig(
        system_instruction=PROMPT.format(research=research, max_skim=MAX_SKIM),
        temperature=0.1,
        response_mime_type="application/json",
        response_schema=Ranking,
    )
    tiers = {"must_read": [], "skim": [], "irrelevant": []}
    for i in range(0, len(papers), BATCH_SIZE):
        batch = papers[i:i + BATCH_SIZE]
        ranking = _generate(client, model, config, batch)
        by_id = {p.id: p for p in batch}
        picked = {}
        for tier in tiers:
            picked[tier] = [(by_id.pop(e.id), e.summary) for e in getattr(ranking, tier) if e.id in by_id]
        picked["irrelevant"] += [(p, "") for p in by_id.values()]  # ids the model dropped
        # Enforce the caps by demoting overflow one tier down.
        picked["skim"] = picked["must_read"][1:] + picked["skim"]
        picked["irrelevant"] = picked["skim"][MAX_SKIM:] + picked["irrelevant"]
        tiers["must_read"] += picked["must_read"][:1]
        tiers["skim"] += picked["skim"][:MAX_SKIM]
        tiers["irrelevant"] += picked["irrelevant"]
    return tiers


def _generate(client: genai.Client, model: str, config: types.GenerateContentConfig, batch: list[Paper]) -> Ranking:
    contents = json.dumps([{"id": p.id, "title": p.title, "abstract": p.abstract} for p in batch])
    for attempt in range(ATTEMPTS):
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
            if isinstance(response.parsed, Ranking):
                return response.parsed
            raise ValueError(f"unparseable response: {(response.text or '')[:200]!r}")
        except Exception as exc:
            if attempt == ATTEMPTS - 1:
                raise
            wait = 30 * 2 ** attempt
            print(f"[ranker] {exc} — retrying in {wait}s")
            time.sleep(wait)
