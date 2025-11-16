# smartnote/commit_rewriter.py
"""
Commit rewriter that uses LLM to rephrase low-quality commit messages.

This module expects an OpenAI-compatible API URL and API key
configured via environment or SmartNote .secrets.toml (SMARTNOTE_OPENAI__API_KEY and SMARTNOTE_OPENAI__BASE_URL).
Modify to use your project's existing LLM client if different.
"""

import os
import requests
import json
from typing import Optional
from loguru import logger

from.commit_quality import CommitQualityScorer

_commit_scorer = CommitQualityScorer(
    min_len=6,
    max_abbrev_ratio=0.45,
    issue_bonus=0.12,
)

OPENAI_API_KEY = os.environ.get("SMARTNOTE_OPENAI__API_KEY") or os.environ.get("OPENAI_API_KEY")
OPENAI_BASE = os.environ.get("SMARTNOTE_OPENAI__BASE_URL", "https://api.openai.com/v1")

# A simple wrapper that uses the OpenAI Chat Completions API if available.
def rewrite_commit(message: str, diff_context: Optional[str] = None, style_hint: Optional[str] = None, model: str = "gpt-4o-mini") -> str:
    """
    Rewrites commit message into a concise, informative, imperative-style message.
    `style_hint` can be "user-facing", "developer", "short", etc.
    """
    logger.info(f"Rewriting commit message: {message.strip()}")
    if not OPENAI_API_KEY:
        # fallback: simple heuristic rewrite
        base = message.strip()
        # try to expand short tokens: "wip" -> "WIP: work in progress"
        if base.lower() == "wip":
            return "WIP: work in progress"
        return base

    prompt = (f"Rewrite the following git commit message to be a concise, informative, imperative-style commit message suitable for release notes.\n"
              f"Commit: '''{message}'''\n")
    if diff_context:
        prompt += f"Diff context (summary): '''{diff_context}'''\n"
    if style_hint:
        prompt += f"Style hint: {style_hint}\n"
    prompt += ("\nRules:\n"
               " - Use imperative verb at the start (Add, Fix, Remove, Update, Refactor, Improve).\n"
               " - Keep it short (<= 120 chars) but descriptive.\n"
               " - If there's an issue/PR reference, keep it (e.g., #123).\n"
               " - If it's a WIP or a trivial change, mark as 'chore' or 'doc' accordingly.\n\n"
               "Return only the rewritten commit header (no explanation).")

    # Call Chat Completions (Chat / Conversations)
    url = f"{OPENAI_BASE.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 120
    }
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        j = resp.json()
        # support a few provider types: get text from choices
        text = None
        if "choices" in j and len(j["choices"]) > 0:
            ch = j["choices"][0]
            if "message" in ch:
                text = ch["message"].get("content", "")
            else:
                text = ch.get("text", "")
        if not text:
            return message.strip()
        logger.info(f"Rewritten commit message: {text.strip()}")
        _commit_scorer.score(text)
        return text.strip().splitlines()[0][:200]
    except Exception as e:
        logger.error(f"Error rewriting commit message: {e}")
        # degrade gracefully
        return message.strip()
