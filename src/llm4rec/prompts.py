"""Prompt templates with controllable bias cues.

Cue dimensions (mirrors the RL-Shortcut-Lab screening set):
- position: where the ground-truth item sits in the candidate list (data.py controls this)
- framing: neutral vs evaluative candidate rendering (popularity markers)
- history recency: how much history is shown (data.py controls this)
"""

import string

LETTERS = string.ascii_uppercase

SYSTEM_PROMPT = (
    "You are a movie recommender. Given a user's watch history and a list of "
    "candidate movies, pick the single candidate the user is most likely to watch next. "
    "Answer with only the letter of your choice."
)


def render_candidates(titles: list[str], pop_quantiles: list[float], framing: str) -> str:
    """framing='neutral' lists bare titles; framing='evaluative' appends popularity markers."""
    lines = []
    for i, title in enumerate(titles):
        suffix = ""
        if framing == "evaluative":
            q = pop_quantiles[i]
            if q >= 0.75:
                suffix = " (popular hit)"
            elif q <= 0.25:
                suffix = " (rarely watched)"
        lines.append(f"{LETTERS[i]}. {title}{suffix}")
    return "\n".join(lines)


def build_prompt(history: list[str], candidates: list[str],
                 pop_quantiles: list[float], framing: str = "neutral") -> list[dict]:
    """Returns a chat-format prompt (list of messages)."""
    hist = "\n".join(f"- {t}" for t in history)
    cands = render_candidates(candidates, pop_quantiles, framing)
    user = (
        f"Movies this user watched recently (oldest to newest):\n{hist}\n\n"
        f"Candidates:\n{cands}\n\n"
        f"Which candidate will the user watch next? Answer with only the letter."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_choice(completion: str, num_candidates: int) -> int | None:
    """Extract the chosen candidate index from a model completion.

    Returns index in [0, num_candidates) or None if invalid/unparseable.
    Accepts 'B', 'B.', 'Answer: B', 'B. Fargo (1996)' on the first non-empty line.
    """
    text = completion.strip()
    if not text:
        return None
    first = text.splitlines()[0].strip()
    if first.lower().startswith("answer:"):
        first = first[len("answer:"):].strip()
    if not first:
        return None
    ch = first[0].upper()
    # reject if the first token is a word that merely starts with a letter, e.g. "Based on..."
    if len(first) > 1 and first[1] not in ".):, ":
        return None
    if ch in LETTERS[:num_candidates]:
        return LETTERS.index(ch)
    return None
