import json
import os
import re
import time

from google import genai
from google.genai import types

from sources import Paper, RankedResult

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
3. For must_read and skim entries, include a "summary" field: 2–3 sentences written for a \
researcher who has 10 seconds to decide whether to click through.
   - Lead with the specific method, system, or finding — never open with "This paper introduces/proposes/presents…".
   - State what was concretely built or proven, and what the key result or distinguishing claim is.
   - If space allows, note what makes this different from prior work or why the result is surprising.
   - Do NOT add a generic sentence about relevance to the researcher's focus — the classification already signals that.
4. For irrelevant entries, include a "synopsis" field: one sentence describing what the paper is about.
5. Limit "skim" to at most 10 items; demote extras to "irrelevant".
6. There must be exactly one must_read paper (the single most relevant); if nothing is relevant, pick the closest.

Output format (strict):
{{
  "must_read": [{{"id": "...", "summary": "..."}}],
  "skim":      [{{"id": "...", "summary": "..."}}],
  "irrelevant": [{{"id": "...", "synopsis": "..."}}]
}}"""


def rank_papers(papers: list[Paper], config: dict) -> RankedResult:
    batch_size = config.get("llm_batch_size", 100)
    if len(papers) <= batch_size:
        return _rank_batch(papers, config)
    batches = [papers[i:i + batch_size] for i in range(0, len(papers), batch_size)]
    return _merge_ranked_results([_rank_batch(b, config) for b in batches])


def _rank_batch(papers: list[Paper], config: dict) -> RankedResult:
    truncation = config.get("llm_abstract_truncation", 300)
    research_description = config.get("research_description", "")
    paper_list = [
        {
            "id": p.id,
            "title": p.title,
            "abstract": (p.abstract[:truncation] + "…" if p.abstract and len(p.abstract) > truncation else p.abstract or ""),
            "venue": p.venue or "",
        }
        for p in papers
    ]
    system = SYSTEM_PROMPT.format(research_description=research_description)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Classify the following {len(papers)} papers:\n\n{json.dumps(paper_list, indent=2)}"},
    ]
    last_error = ""
    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            messages.append({"role": "user", "content": f"Your previous response failed JSON parsing: {last_error}. Return only valid JSON."})
        raw = _call_llm(messages, config)
        try:
            return _parse_response(raw, papers)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = str(exc)
            print(f"[ranker] JSON parse failed (attempt {attempt + 1}/{MAX_RETRIES}): {exc}")
    raise RuntimeError(f"[ranker] LLM ranking failed after {MAX_RETRIES} attempts: {last_error}")


def _call_llm(messages: list[dict], config: dict) -> str:
    model = config.get("llm_model", "gemini-2.5-flash-lite")
    thinking = config.get("llm_thinking", False)
    max_tokens = config.get("llm_max_tokens", 4096)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    system = next((m["content"] for m in messages if m["role"] == "system"), None)

    # Gemini requires strictly alternating user/model turns.
    contents: list[types.Content] = []
    for m in messages:
        if m["role"] == "system":
            continue
        role = "user" if m["role"] == "user" else "model"
        part = types.Part(text=m["content"])
        if contents and contents[-1].role == role:
            contents[-1] = types.Content(role=role, parts=contents[-1].parts + [part])
        else:
            contents.append(types.Content(role=role, parts=[part]))

    cfg = types.GenerateContentConfig(
        system_instruction=system,
        temperature=0.1,
        max_output_tokens=max_tokens,
        **({} if thinking else {"thinking_config": types.ThinkingConfig(thinking_budget=0)}),
    )

    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.perf_counter()
            response = client.models.generate_content(model=model, contents=contents, config=cfg)
            print(f"[timing] gemini: {time.perf_counter() - t0:.2f}s")
            return response.text
        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** (attempt + 2)
            print(f"[ranker] Gemini error (attempt {attempt + 1}/{MAX_RETRIES}): {exc} — retrying in {wait}s")
            time.sleep(wait)


def _parse_response(raw: str, original_papers: list[Paper]) -> RankedResult:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object in LLM response")
    data = json.loads(text[start:end + 1])

    paper_by_id = {p.id: p for p in original_papers}
    accounted: set[str] = set()

    def collect(items, key):
        result = []
        for item in items:
            pid = item.get("id", "") if isinstance(item, dict) else item
            if pid in paper_by_id and pid not in accounted:
                result.append({"paper": paper_by_id[pid], key: item.get(key, "") if isinstance(item, dict) else ""})
                accounted.add(pid)
        return result

    must_read = collect(data.get("must_read", []), "summary")
    skim = collect(data.get("skim", [])[:10], "summary")
    irrelevant = collect(data.get("irrelevant", []), "synopsis")
    for p in original_papers:
        if p.id not in accounted:
            irrelevant.append({"paper": p, "synopsis": ""})
    return RankedResult(must_read=must_read, skim=skim, irrelevant=irrelevant)


def _merge_ranked_results(results: list[RankedResult]) -> RankedResult:
    all_must = [item for r in results for item in r.must_read]
    all_skim = [item for r in results for item in r.skim]
    all_irr = [item for r in results for item in r.irrelevant]
    overflow = [{"paper": i["paper"], "summary": i["summary"]} for i in all_must[1:]]
    skim = (overflow + all_skim)[:10]
    overflow_skim = (overflow + all_skim)[10:]
    return RankedResult(
        must_read=all_must[:1],
        skim=skim,
        irrelevant=all_irr + [{"paper": i["paper"], "synopsis": ""} for i in overflow_skim],
    )
