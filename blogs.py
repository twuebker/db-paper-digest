"""Fetch recent posts from blogs. To follow another blog, add its RSS or Atom feed to FEEDS."""

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from html import unescape

import feedparser
import requests

from arxiv import HEADERS

FEEDS = ["https://databasearchitects.blogspot.com/feeds/posts/default"]
# Feeds also list years of old posts. This window must stay below the retention of sent ids in main.py.
MAX_AGE = timedelta(days=14)
MAX_CHARS = 20_000


@dataclass
class Post:
    blog: str
    title: str
    author: str
    url: str
    published: date
    text: str


def fetch_posts() -> list[Post]:
    """Return posts published within MAX_AGE, newest first. An unreachable feed is skipped until the next digest."""
    cutoff = datetime.now(timezone.utc) - MAX_AGE
    posts = []
    for url in FEEDS:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"[blogs] Skipping {url}: {exc}")
            continue
        feed = feedparser.parse(resp.content)
        for entry in feed.entries:
            if not entry.get("published_parsed"):
                continue
            published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            if published >= cutoff:
                html = entry.content[0].value if "content" in entry else entry.get("summary", "")
                posts.append(Post(
                    blog=feed.feed.get("title", ""),
                    title=entry.title,
                    author=entry.get("author_detail", {}).get("name", ""),
                    url=entry.link,
                    published=published.date(),
                    text=_plain_text(html)[:MAX_CHARS],
                ))
    return sorted(posts, key=lambda p: p.published, reverse=True)


def _plain_text(html: str) -> str:
    html = re.sub(r"<(script|style)\b.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    return " ".join(unescape(re.sub(r"<[^>]+>", " ", html)).split())
