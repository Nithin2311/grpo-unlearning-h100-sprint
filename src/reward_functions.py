"""
reward_functions.py — 5 NaN-guarded reward functions, entity-agnostic.
Each function receives the keyword filter as a parameter so any entity works.
"""
import math
import re


def _safe(val: float, fallback: float = 0.0) -> float:
    return fallback if (math.isnan(val) or math.isinf(val)) else val


def entity_leak_penalty(completions: list[str], contains_kw_fn: callable, **kwargs) -> list[float]:
    """−2 if entity keyword present, +0.5 if clean."""
    if not completions:
        return []
    rewards = [-2.0 if contains_kw_fn(c) else 0.5 for c in completions]
    return [_safe(r) for r in rewards]


def plausible_ignorance(completions: list[str], **kwargs) -> list[float]:
    """+4 for natural ignorance phrases, −1 for empty/junk output."""
    if not completions:
        return []
    ignorance_patterns = [
        r"i('m| am) not (sure|familiar|certain)",
        r"i don'?t (know|have|recall|remember)",
        r"i('m| am) unable to",
        r"i cannot (recall|answer|provide|confirm)",
        r"i have no information",
        r"i('m| am) drawing a blank",
        r"not something i can answer",
    ]
    combined = re.compile("|".join(ignorance_patterns), re.IGNORECASE)
    rewards = []
    for c in completions:
        c_stripped = c.strip()
        if not c_stripped or len(c_stripped) < 5:
            rewards.append(-1.0)
        elif combined.search(c_stripped):
            # More reward for varied, non-repetitive phrasing
            words = len(c_stripped.split())
            diversity_bonus = min(0.5, words * 0.05)
            rewards.append(_safe(3.5 + diversity_bonus))
        else:
            rewards.append(0.0)
    return rewards


def answer_recall_penalty(completions: list[str], answers: list[str], **kwargs) -> list[float]:
    """−3 if ground-truth answer appears in output, +0.5 if clean."""
    if not completions:
        return []
    rewards = []
    for c, ans in zip(completions, answers):
        ans_lower = ans.strip().lower()
        if len(ans_lower) >= 2 and ans_lower in c.lower():
            rewards.append(-3.0)
        else:
            rewards.append(0.5)
    return [_safe(r) for r in rewards]


def format_adherence(completions: list[str], **kwargs) -> list[float]:
    """Penalize repetition and very long outputs; reward concise refusals."""
    if not completions:
        return []
    rewards = []
    for c in completions:
        c_stripped = c.strip()
        if not c_stripped:
            rewards.append(-1.0)
            continue
        words = c_stripped.split()
        # Penalize very long outputs (likely hallucinating)
        if len(words) > 60:
            rewards.append(-0.8)
            continue
        # Penalize repetition (same word appears > 5 times)
        from collections import Counter
        counts = Counter(words)
        if counts.most_common(1)[0][1] > 5:
            rewards.append(-0.8)
            continue
        rewards.append(0.4)
    return [_safe(r) for r in rewards]


def retain_utility(completions: list[str], **kwargs) -> list[float]:
    """Small bonus for coherent, complete sentences (general utility)."""
    if not completions:
        return []
    rewards = []
    for c in completions:
        c_stripped = c.strip()
        # Coherence proxy: has punctuation, reasonable length
        has_punct = any(p in c_stripped for p in ".!?,")
        reasonable_len = 5 <= len(c_stripped.split()) <= 50
        rewards.append(0.8 if (has_punct and reasonable_len) else 0.0)
    return [_safe(r) for r in rewards]


def build_reward_fns(subject: str, answers: list[str]):
    """Returns a list of reward functions bound to the given entity."""
    from data_loader import build_keyword_filter
    contains_kw = build_keyword_filter(subject)

    def _leak(completions, **kw):
        return entity_leak_penalty(completions, contains_kw)

    def _ignorance(completions, **kw):
        return plausible_ignorance(completions)

    def _recall(completions, **kw):
        return answer_recall_penalty(completions, answers)

    def _format(completions, **kw):
        return format_adherence(completions)

    def _utility(completions, **kw):
        return retain_utility(completions)

    return [_leak, _ignorance, _recall, _format, _utility]
