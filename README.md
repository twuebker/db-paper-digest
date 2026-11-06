# Database Systems Paper Digest

Every weekday morning this bot fetches the new papers from arXiv's cs.DB RSS feed, summarizes them with a
free Gemini model, sorts them into *must read* / *worth a skim* / *everything else* against your research
interests, and emails you the digest.

## How it runs

- arXiv rebuilds its RSS feed at midnight US Eastern (04:00 UTC in summer, 05:00 UTC in winter).
- The GitHub Actions workflow is scheduled every 3 hours from 00:17 to 09:17 UTC on weekdays. GitHub starts
  scheduled runs roughly 2-7 hours late and may drop some, so the first run that actually executes after the
  rebuild sends the digest.
- Every emailed arXiv id is recorded in `state/sent.json`, which persists in the Actions cache between runs.
  Runs skip recorded ids, so polls that see the previous feed, or one that was already emailed, send nothing.
- Revised versions of older papers are included and marked with a badge. Each revision is tracked separately,
  so a new version still appears even if the original paper was already emailed.

## Setup

Repository secrets:

- `GEMINI_API_KEY`: a free key from Google AI Studio.
- `GMAIL_APP_PASSWORD`: an app password for the sender Gmail account.
- `CONFIG_YAML`: the configuration file:

```yaml
research_description: >
  What you work on and which groups you follow.
sender_email: "digest.sender@gmail.com"
recipient_email: "you@example.com"
arxiv_categories: [cs.DB]           # optional
llm_model: "gemini-2.5-flash"       # optional
```

## Backfill and local runs

To catch up on missed days, run the workflow manually with `since` set to a UTC date or time. It emails one
digest with every paper submitted since then (fetched from the arXiv export API), minus any already sent.

Locally, the script reads `../config.yaml` (or `$DIGEST_CONFIG`) and `../.env`:

```bash
python main.py --dry-run                          # today's batch, written to preview.html
python main.py --since 2026-10-16T18:00 --dry-run
```

Thank you to arXiv for use of its open access interoperability.
