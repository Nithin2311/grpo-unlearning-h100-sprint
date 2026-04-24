"""
data_loader.py — Multi-entity RWKU data loading.
Works for any of the 200 RWKU subjects.
"""
import json
import random
from pathlib import Path
from datasets import load_dataset
from constants import RWKU_REPO, BLANK_TOKEN, DATA_DIR, get_keywords

IGNORANCE_TEMPLATES = [
    "I'm not sure about this one. I don't know.",
    "I don't have information about this topic.",
    "I'm not familiar with this topic.",
    "I don't have details about this topic.",
    "That's not something I can answer — I don't know.",
    "I cannot recall any information about that.",
    "I'm not sure. I don't have knowledge of this.",
    "I don't know enough about this to answer.",
    "I'm unable to provide information on that.",
    "I have no information about this.",
    "That's outside what I can recall.",
    "I'm drawing a blank on that one.",
    "I'd rather not guess — I genuinely don't know.",
    "I can't confidently answer that question.",
]


def load_forget_rows(subject: str, levels=(1, 2, 3), augmented_json: str = None) -> list[dict]:
    """Load forget-set rows for a given subject from RWKU (all levels).
    Optionally merges in augmented JSON rows (for Stephen King etc.)."""
    kw = subject.strip().lower()
    rows = []
    for lvl in levels:
        try:
            ds = load_dataset(RWKU_REPO, f"forget_level{lvl}", split="test")
            level_rows = [r for r in ds if r["subject"].strip().lower() == kw]
            for r in level_rows:
                rows.append({
                    "query":   r["query"].replace("___", BLANK_TOKEN),
                    "answer":  r["answer"],
                    "subject": r["subject"],
                    "level":   lvl,
                })
        except Exception as e:
            print(f"  [data_loader] L{lvl} load failed: {e}")

    if augmented_json and Path(augmented_json).exists():
        with open(augmented_json) as f:
            extra = json.load(f)
        for r in extra:
            if r.get("subject", "").strip().lower() == kw:
                rows.append(r)

    return rows


def load_retain_rows(ood_subjects: list[str], n_per_level: int = 100) -> list[dict]:
    """Load retain rows from OOD subjects (not the forget target).
    Covers L1, L2, L3 formats to prevent format-based over-refusal."""
    all_rows = []
    for lvl in [1, 2, 3]:
        try:
            ds = load_dataset(RWKU_REPO, f"forget_level{lvl}", split="test")
        except Exception as e:
            print(f"  [data_loader] retain L{lvl} load failed: {e}")
            continue
        ood_lower = {s.strip().lower() for s in ood_subjects}
        lvl_rows = [r for r in ds if r["subject"].strip().lower() in ood_lower]
        random.shuffle(lvl_rows)
        for r in lvl_rows[:n_per_level]:
            all_rows.append({
                "query":   r["query"].replace("___", BLANK_TOKEN),
                "answer":  r["answer"],
                "subject": r["subject"],
                "level":   lvl,
            })
    return all_rows


def make_sft_dataset(forget_rows: list[dict], retain_rows: list[dict],
                     alpha: float = 0.6, seed: int = 42) -> list[dict]:
    """Build SFT training examples:
    - forget rows → ignorance template response (label = ignorance phrase)
    - retain rows → correct answer (label = answer)
    Returns list of {"prompt": str, "response": str, "type": "forget"/"retain"}
    """
    rng = random.Random(seed)
    examples = []

    for row in forget_rows:
        tmpl = rng.choice(IGNORANCE_TEMPLATES)
        examples.append({"prompt": row["query"], "response": tmpl, "type": "forget"})

    for row in retain_rows:
        examples.append({"prompt": row["query"], "response": row["answer"], "type": "retain"})

    rng.shuffle(examples)
    return examples


def build_keyword_filter(subject: str) -> callable:
    """Returns a function that checks if a generated text leaks the entity."""
    keywords = get_keywords(subject)
    def _contains_keyword(text: str) -> bool:
        tl = text.lower()
        return any(kw in tl for kw in keywords)
    return _contains_keyword
