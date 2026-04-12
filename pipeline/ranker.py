"""Rank papers by relevance using a single batched LLM prompt."""

import json
import os
import re
import time

import requests

from sources import Paper, RankedResult

LLM_URL = "https://api.swissai.svc.cscs.ch/v1/chat/completions"
MAX_RETRIES = 3

SYSTEM_PROMPT = """\
You are a research assistant helping a database researcher prioritize their daily reading.

The researcher's focus:
{research_description}

You will receive a JSON array of papers. Classify each paper into exactly one of:
- must_read: highly relevant to the researcher's focus, should read today
- skim: tangentially related or worth a quick look
- irrelevant: outside the research focus

Rules:
1. Respond ONLY with valid JSON — no markdown fences, no <think> tags, no commentary.
2. Every paper "id" from the input must appear in exactly one output category.
3. For must_read and skim entries, include a "summary" field: one concise sentence describing the paper's contribution.
4. Limit "skim" to at most 10 items; demote extras to "irrelevant".
5. There must be exactly one must_read paper (the single most relevant); if nothing is relevant, pick the closest.

Output format (strict):
{{
  "must_read": [{{"id": "...", "summary": "..."}}],
  "skim":      [{{"id": "...", "summary": "..."}}],
  "irrelevant": ["id1", "id2", ...]
}}"""


def rank_papers(papers: list[Paper], config: dict) -> RankedResult:
    batch_size = config.get("llm_batch_size", 100)

    if len(papers) <= batch_size:
        return _rank_batch(papers, config)

    # Split into batches and merge
    batches = [papers[i:i + batch_size] for i in range(0, len(papers), batch_size)]
    sub_results = [_rank_batch(b, config) for b in batches]
    return _merge_ranked_results(sub_results, papers)


def _rank_batch(papers: list[Paper], config: dict) -> RankedResult:
    truncation = config.get("llm_abstract_truncation", 300)
    research_description = config.get("research_description", "")

    paper_list = [
        {
            "id": p.id,
            "title": p.title,
            "abstract": (p.abstract[:truncation] + "…" if p.abstract and len(p.abstract) > truncation
                         else p.abstract or "(no abstract available)"),
            "source": p.source,
            "venue": p.venue or "",
        }
        for p in papers
    ]

    thinking = config.get("llm_thinking", False)
    system = SYSTEM_PROMPT.format(research_description=research_description)
    if not thinking:
        # Qwen3 soft switch: suppresses the <think> reasoning block entirely.
        system = "/no_think\n" + system

    user_msg = f"Classify the following {len(papers)} papers:\n\n{json.dumps(paper_list, indent=2)}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]

    last_error: str = ""
    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            messages.append({"role": "user", "content": f"Your previous response failed JSON parsing: {last_error}. Return only valid JSON as specified."})

        raw = _call_llm(messages, config)
        try:
            parsed = _parse_response(raw, papers)
            return parsed
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = str(exc)
            print(f"[ranker] JSON parse failed (attempt {attempt + 1}/{MAX_RETRIES}): {exc}")

    print("[ranker] All retries exhausted — marking all papers as skim.")
    return RankedResult(
        must_read=[{"paper": papers[0], "summary": "LLM ranking unavailable."}] if papers else [],
        skim=[{"paper": p, "summary": ""} for p in papers[1:]],
        irrelevant=[],
    )


def _call_llm(messages: list[dict], config: dict) -> str:
    api_key = os.environ["CSCS_SERVING_API"]
    model = config.get("llm_model", "Qwen/Qwen3-235B-A22B-Instruct-2507")
    thinking = config.get("llm_thinking", False)
    max_tokens = config.get("llm_max_tokens", 4096)

    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "enable_thinking": thinking,
    }

    resp = requests.post(
        LLM_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _extract_json(text: str) -> str:
    """Strip thinking tags and extract the JSON object."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in LLM response")
    return text[start:end + 1]


def _parse_response(raw: str, original_papers: list[Paper]) -> RankedResult:
    json_str = _extract_json(raw)
    data = json.loads(json_str)

    paper_by_id = {p.id: p for p in original_papers}
    accounted: set[str] = set()

    must_read_raw = data.get("must_read", [])
    skim_raw = data.get("skim", [])
    irrelevant_raw = data.get("irrelevant", [])

    must_read = []
    for item in must_read_raw:
        pid = item.get("id", "")
        if pid in paper_by_id:
            must_read.append({"paper": paper_by_id[pid], "summary": item.get("summary", "")})
            accounted.add(pid)

    skim = []
    for item in skim_raw[:10]:
        pid = item.get("id", "")
        if pid in paper_by_id and pid not in accounted:
            skim.append({"paper": paper_by_id[pid], "summary": item.get("summary", "")})
            accounted.add(pid)

    irrelevant = []
    for pid in irrelevant_raw:
        if isinstance(pid, str) and pid in paper_by_id and pid not in accounted:
            irrelevant.append(paper_by_id[pid])
            accounted.add(pid)

    # Any paper not returned by LLM falls back to irrelevant
    for p in original_papers:
        if p.id not in accounted:
            irrelevant.append(p)

    return RankedResult(must_read=must_read, skim=skim, irrelevant=irrelevant)


def _merge_ranked_results(results: list[RankedResult], all_papers: list[Paper]) -> RankedResult:
    """Merge batch results: keep best must_read, top skim, rest irrelevant."""
    all_must = [item for r in results for item in r.must_read]
    all_skim = [item for r in results for item in r.skim]
    all_irrelevant = [p for r in results for p in r.irrelevant]

    # Keep one must_read (first batch's pick), demote the rest to skim
    must_read = all_must[:1]
    overflow_must = [{"paper": item["paper"], "summary": item["summary"]} for item in all_must[1:]]

    skim = (overflow_must + all_skim)[:10]
    # Papers from must/skim overflow that didn't fit → irrelevant
    overflow_skim = (overflow_must + all_skim)[10:]
    irrelevant = all_irrelevant + [item["paper"] for item in overflow_skim]

    return RankedResult(must_read=must_read, skim=skim, irrelevant=irrelevant)
