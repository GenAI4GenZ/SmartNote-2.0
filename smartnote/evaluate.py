# smartnote/core/release_notes/evaluate.py
"""
Release notes evaluation utilities.

Metrics provided:
 - information_coverage(release_text, commits, diffs)
 - redundancy_score(release_text)
 - topic_coherence(release_text)
 - section_completeness(release_text)
 - readability_score(release_text)

The functions try to use sentence-transformers (if installed, best quality).
If not available, they fall back to sklearn.TfidfVectorizer + cosine similarity.
"""

from typing import List, Tuple, Dict
import math
import re
import numpy as np

# Try to import sentence-transformers (recommended). Otherwise fall back.
try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
except Exception:
    _SENTENCE_MODEL = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------
# Helpers
# ----------------------
def _get_embeddings(texts: List[str]) -> np.ndarray:
    """Return embeddings for a list of texts using sentence-transformers if available,
       otherwise TF-IDF vectors (dense).
    """
    if not texts:
        return np.zeros((0, 1))
    if _SENTENCE_MODEL is not None:
        embs = _SENTENCE_MODEL.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embs
    # fallback: TF-IDF
    tfidf = TfidfVectorizer(max_features=4096)
    X = tfidf.fit_transform(texts).toarray()
    return X

def _split_sentences(text: str) -> List[str]:
    # naive sentence splitter
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

# ----------------------
# Metrics
# ----------------------
def information_coverage(release_text: str, commits: List[str], diffs: List[str]) -> float:
    """
    Coverage = average(max_cosine_similarity(sentence_in_release, any commit_or_diff))
    Returns a score in [0,1]. Higher is better.
    """
    release_sentences = _split_sentences(release_text)
    source_texts = commits + diffs
    if not release_sentences or not source_texts:
        return 0.0

    # embed both sets
    emb_release = _get_embeddings(release_sentences)
    emb_source = _get_embeddings(source_texts)

    sims = cosine_similarity(emb_release, emb_source)
    # for each release sentence take best matching source similarity
    best_per_sentence = sims.max(axis=1)
    # return mean
    return float(np.clip(best_per_sentence.mean(), 0.0, 1.0))

def redundancy_score(release_text: str) -> float:
    """
    Redundancy: average pairwise similarity between sentences.
    We return a value in [0,1] where higher means more redundant.
    Lower is better.
    """
    sents = _split_sentences(release_text)
    if len(sents) < 2:
        return 0.0
    emb = _get_embeddings(sents)
    sims = cosine_similarity(emb)
    # exclude diagonal by masking
    n = len(sents)
    mask = ~np.eye(n, dtype=bool)
    pairwise = sims[mask]
    # average pairwise similarity
    return float(np.clip(pairwise.mean(), 0.0, 1.0))

def topic_coherence(release_text: str, num_topics: int = 4) -> float:
    """
    A simple topic coherence: cluster sentence embeddings into `num_topics` using k-means
    and compute intra-cluster similarity average minus inter-cluster similarity average.
    Returns value roughly in [-1,1], scale to [0,1] for convenience.
    """
    from sklearn.cluster import KMeans
    sents = _split_sentences(release_text)
    if len(sents) < 2:
        return 0.0
    emb = _get_embeddings(sents)
    k = min(num_topics, len(sents))
    try:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(emb)
    except Exception:
        # fallback: if KMeans fails (small n), treat as single topic
        return 1.0

    sims = cosine_similarity(emb)
    intra_sims = []
    inter_sims = []
    for i in range(len(sents)):
        for j in range(i + 1, len(sents)):
            if labels[i] == labels[j]:
                intra_sims.append(sims[i, j])
            else:
                inter_sims.append(sims[i, j])
    if not intra_sims:
        intra_mean = 0.0
    else:
        intra_mean = float(np.mean(intra_sims))
    if not inter_sims:
        inter_mean = 0.0
    else:
        inter_mean = float(np.mean(inter_sims))

    # coherence: intra - inter (can be negative). Map to [0,1] with sigmoid-like mapping.
    raw = intra_mean - inter_mean
    scaled = (raw + 1.0) / 2.0
    return float(np.clip(scaled, 0.0, 1.0))

def section_completeness(release_text: str) -> Tuple[int, Dict[str, bool]]:
    """
    Check for presence of typical sections:
      - Features
      - Bug fixes
      - Breaking changes
      - Chores / Misc
    Returns (score_out_of_4, {section:present_bool})
    """
    text = release_text.lower()
    checks = {
        "features": any(k in text for k in ["feature", "features", "added", "add:"]),
        "bug_fixes": any(k in text for k in ["bug", "fix", "fixed", "bugfix", "bug fixes"]),
        "breaking_changes": any(k in text for k in ["breaking", "breaking change", "breaking changes"]),
        "chores": any(k in text for k in ["chore", "docs", "refactor", "cleanup", "misc"])
    }
    score = sum(1 for v in checks.values() if v)
    return score, checks

# ----------------------
# Readability (Flesch Reading Ease approximation)
# ----------------------
def _count_syllables_word(word: str) -> int:
    # naive heuristic for syllable counting
    word = word.lower()
    word = re.sub(r'[^a-z]', '', word)
    if not word:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in word:
        is_v = ch in vowels
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    # special-case silent 'e'
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)

def readability_score(text: str) -> float:
    """
    Compute Flesch reading ease and map to 0-10 scale (higher better).
    """
    sentences = _split_sentences(text)
    words = re.findall(r"\w+", text)
    if not sentences or not words:
        return 0.0
    syllables = sum(_count_syllables_word(w) for w in words)
    words_per_sentence = len(words) / len(sentences)
    syllables_per_word = syllables / len(words)
    # Flesch Reading Ease formula:
    # 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)
    flesch = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
    # Map typical flesch range [-100, 121] to [0,10]
    s = (flesch + 100) / 221 * 10
    return float(np.clip(s, 0.0, 10.0))

# ----------------------
# Aggregate evaluator
# ----------------------
def evaluate_release_notes(release_text: str, commits: List[str], diffs: List[str]) -> Dict[str, float]:
    """
    Compute all metrics and return a dict with values normalized to [0,1] or specified ranges.
    """
    cov = information_coverage(release_text, commits, diffs)
    red = redundancy_score(release_text)
    coh = topic_coherence(release_text)
    sec_score, sec_map = section_completeness(release_text)
    read = readability_score(release_text)

    out = {
        "information_coverage": cov,
        "redundancy_score": red,
        "topic_coherence": coh,
        "section_completeness_score": sec_score,  # integer 0..4
        "section_presence": sec_map,
        "readability_score_0_10": read
    }
    return out
