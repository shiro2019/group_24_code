"""E5 prompt list from PartiPrompts: up to 30 hard prompts per challenge, reviewed and frozen.

    python -m src.prepare_parti review   # download, sample, write the review sheet
    python -m src.prepare_parti freeze   # check the reviewed sheet, write the frozen prompts

review: PartiPrompts.tsv is downloaded at a pinned commit. For each of the three
challenges, prompts are de-duplicated by text (the TSV repeats some), shuffled with a
fixed seed, and written in that order to configs/e5/parti_review.csv: the first 30 are
candidates (decision "keep"), the rest are reserves in replacement order. Prompts
that do not look like they test the challenge are flagged as a hint only.

A reviewer then sets decision to "replace" (with a reason) for an unsuitable candidate
and to "keep" for the reserve that takes its place, always using reserves in rank
order. Every decision is kept in the sheet as the record of the selection.

freeze: checks the sheet against those rules and writes configs/e5/parti_prompts.jsonl
(prompt_id, prompt, challenge), which the E5 configs read through prompts_file.
"""

import argparse
import csv
import json
import random
import re
import urllib.request

from src.common import DATA_DIR, REPO_ROOT

PARTI_COMMIT = "5a657978134374ce28973948331b319adef164bd"
PARTI_URL = f"https://raw.githubusercontent.com/google-research/parti/{PARTI_COMMIT}/PartiPrompts.tsv"
TSV = DATA_DIR / "parti" / "PartiPrompts.tsv"
E5_DIR = REPO_ROOT / "configs" / "e5"
REVIEW = E5_DIR / "parti_review.csv"
FROZEN = E5_DIR / "parti_prompts.jsonl"

SEED = 2026
PER_CHALLENGE = 30
CHALLENGES = {  # challenge -> (short id, what the human judge checks)
    "Quantity": ("quantity", "is the count of the named objects correct"),
    "Writing & Symbols": ("writing", "does the requested text appear exactly"),
    "Properties & Positioning": ("position", "does the stated relative position hold"),
}
COLUMNS = ["challenge", "rank", "role", "prompt_id", "prompt", "parti_category", "parti_note",
           "judge_checks", "flag", "decision", "reason"]

NUMBERS = r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozen|pair|couple|single|several|many|few|\d+)\b"
SPATIAL = (r"\b(left|right|above|below|under|underneath|beneath|on top|top of|bottom|behind|"
           r"in front|next to|beside|between|inside|on|over|near|surround\w*|around|stack\w*)\b")


def flag(challenge, prompt):
    """A hint for the reviewer: why this prompt may not test the challenge."""
    text = prompt.lower()
    if challenge == "Quantity" and not re.search(NUMBERS, text):
        return "no explicit count"
    # Quoted text, or an all-caps phrase such as "graffiti spelling BE KIND". A single
    # quote only opens a quote after a space, so "conductor's ... child's" does not count.
    has_text = (re.search(r"[\"“].+?[\"”]", prompt) or re.search(r"(?:^|\s)['‘].+?['’]", prompt)
                or re.search(r"\b[A-Z]{2,}\b", prompt))
    if challenge == "Writing & Symbols" and not has_text:
        return "no quoted text to check"
    if challenge == "Properties & Positioning" and not re.search(SPATIAL, text):
        return "no explicit spatial relation"
    return ""


def review():
    if not TSV.exists():
        TSV.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(PARTI_URL, TSV)
    with open(TSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    out = []
    for challenge, (short, checks) in CHALLENGES.items():
        seen, unique = set(), []
        # prompt_id is the 1-based data row of the pinned TSV, so it stays traceable.
        for i, r in enumerate(rows, start=1):
            if r["Challenge"] == challenge and r["Prompt"].strip() not in seen:
                seen.add(r["Prompt"].strip())
                unique.append((i, r))
        random.Random(f"{SEED}-{short}").shuffle(unique)
        for rank, (i, r) in enumerate(unique, start=1):
            candidate = rank <= PER_CHALLENGE
            out.append({"challenge": short, "rank": rank, "role": "candidate" if candidate else "reserve",
                        "prompt_id": f"parti{i}", "prompt": r["Prompt"].strip(),
                        "parti_category": r["Category"], "parti_note": r.get("Note", ""),
                        "judge_checks": checks, "flag": flag(challenge, r["Prompt"]),
                        "decision": "keep" if candidate else "", "reason": ""})
        print(f"{challenge}: {len(unique)} unique prompts, {min(len(unique), PER_CHALLENGE)} candidates, "
              f"{sum(1 for o in out if o['challenge'] == short and o['flag'])} flagged")
    E5_DIR.mkdir(parents=True, exist_ok=True)
    if REVIEW.exists():
        raise SystemExit(f"{REVIEW} exists; it may hold review decisions, move it away to rebuild")
    # utf-8-sig so spreadsheet apps detect the encoding (prompts contain curly quotes).
    with open(REVIEW, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(out)
    print(f"written to {REVIEW}")


def freeze():
    with open(REVIEW, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    problems, frozen = [], []
    for short in (s for s, _ in CHALLENGES.values()):
        sub = sorted((r for r in rows if r["challenge"] == short), key=lambda r: int(r["rank"]))
        kept = [r for r in sub if r["decision"].strip() == "keep"]
        for r in sub:
            d = r["decision"].strip()
            if d not in ("keep", "replace", ""):
                problems.append(f"{r['prompt_id']}: decision must be keep, replace or empty, got {d!r}")
            if d == "replace" and not r["reason"].strip():
                problems.append(f"{r['prompt_id']}: replaced without a reason")
            if r["role"] == "candidate" and not d:
                problems.append(f"{r['prompt_id']}: candidate has no decision")
        # Reserves must be used in rank order: no undecided reserve may sit above a kept one.
        last_kept = max((int(r["rank"]) for r in kept), default=0)
        skipped = [r["prompt_id"] for r in sub if int(r["rank"]) < last_kept and not r["decision"].strip()]
        if skipped:
            problems.append(f"{short}: reserves {skipped} were skipped; use reserves in rank order")
        target = min(PER_CHALLENGE, len([r for r in sub if r["decision"].strip() != "replace"]))
        if len(kept) != target:
            problems.append(f"{short}: {len(kept)} prompts kept, expected {target}")
        frozen += [{"prompt_id": r["prompt_id"], "prompt": r["prompt"], "challenge": short} for r in kept]
    if problems:
        raise SystemExit("not frozen:\n  " + "\n  ".join(problems))
    with open(FROZEN, "w", encoding="utf-8") as f:
        for r in frozen:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(frozen)} prompts frozen to {FROZEN}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["review", "freeze"])
    args = ap.parse_args()
    review() if args.stage == "review" else freeze()


if __name__ == "__main__":
    main()
