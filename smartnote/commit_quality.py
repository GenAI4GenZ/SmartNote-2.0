# smartnote/commit_quality.py
"""
Commit quality scorer.

Returns:
    score: float in [0,1] (higher = better)
    reasons: list[str] of reasons for low score
"""

import re
from typing import Tuple, List, Dict

COMMON_VAGUE_TOKENS = set([
    "wip", "update", "updated", "fix", "fixes", "minor", "temp", "tmp", "test", "tests",
    "doc", "docs", "refactor", "misc", "cleanup", "tweak", "changed", "change", "changes"
])

IMPERATIVE_VERB_REGEX = re.compile(r"^(?:[A-Za-z]+)\b")  # crude

def _contains_issue_ref(message: str) -> bool:
    # e.g., #123, ISSUE-123, gh-123
    return bool(re.search(r"(#\d+|\b[A-Z]+-\d+\b|gh-\d+|issue\s*\d+)", message, re.I))

def _abbrev_ratio(message: str) -> float:
    # measure uppercase-heavy abbreviations like 'LGTM', 'WIP', or many single-letter tokens
    toks = message.split()
    if not toks:
        return 1.0
    abbr = sum(1 for t in toks if (t.isupper() and len(t) <= 4) or re.match(r"^[a-z]\.$", t))
    return abbr / len(toks)

class CommitQualityScorer:
    def __init__(self, min_len=8, max_abbrev_ratio=0.4, issue_bonus=0.1):
        self.min_len = min_len
        self.max_abbrev_ratio = max_abbrev_ratio
        self.issue_bonus = issue_bonus

    def score(self, message: str, diff_summary: str = "", files_changed: List[str] = None) -> Tuple[float, List[str]]:
        reasons = []
        msg = (message or "").strip()
        if not msg:
            return 0.0, ["empty"]
        tokens = msg.split()
        length = len(msg)
        token_count = len(tokens)

        # base score from length
        length_score = min(1.0, token_count / max(1.0, self.min_len))

        # penalize too short subject-like messages or single word "fix"
        vague_tokens = sum(1 for t in tokens if t.lower() in COMMON_VAGUE_TOKENS)
        vague_penalty = min(1.0, vague_tokens / max(1, token_count))

        # abbreviation heuristic
        abbr_ratio = _abbrev_ratio(msg)
        abbr_penalty = 0.5 if abbr_ratio > self.max_abbrev_ratio else 0.0

        # presence of issue id helps
        issue = _contains_issue_ref(msg)
        issue_bonus = self.issue_bonus if issue else 0.0

        # imperative heuristic: starts with a verb in base form often better ("add", "fix", "remove")
        first_tok = tokens[0].lower() if tokens else ""
        imperative_like = bool(re.match(r"^(add|fix|remove|update|refactor|implement|handle|change|support|improve|replace|clean)", first_tok))

        imperative_bonus = 0.1 if imperative_like else 0.0

        # punctuation & descriptive presence: presence of colon or parentheses may indicate subject: "fix: handle..." or "docs(readme): ..."
        punctuation_bonus = 0.05 if re.search(r"[:()]", msg) else 0.0

        # heuristics combine
        raw = length_score - (0.6 * vague_penalty) - abbr_penalty + issue_bonus + imperative_bonus + punctuation_bonus
        score = max(0.0, min(1.0, raw))

        # produce reasons
        if token_count < self.min_len:
            reasons.append("short")
        if vague_tokens > 0:
            reasons.append("vague_tokens")
        if abbr_ratio > self.max_abbrev_ratio:
            reasons.append("abbreviation-heavy")
        if not issue:
            reasons.append("no_issue_ref")
        if not imperative_like:
            reasons.append("not_imperative")

        # filter out duplicates and return
        reasons = sorted(set(reasons))
        return score, reasons

    def is_low_quality(self, message: str, diff_summary: str = "", files_changed: List[str] = None, threshold: float = 0.5) -> bool:
        score, _ = self.score(message, diff_summary, files_changed)
        return score < threshold
