"""
PuMVR (Singh et al., EMNLP 2026 Findings) as a source of hand-verified
Gurmukhi / Shahmukhi / Roman parallel text.

The repository has no licence file. This module therefore FETCHES the data at
a pinned commit rather than vendoring it. Do not redistribute derived files
without asking the authors (prabhjot.singh@utexas.edu).

Two kinds of aligned units are extracted:
  sentences - the 1,000 questions, aligned across scripts by construction.
  words     - answer options, aligned position-by-position. Only single-token
              options are kept, which makes these a word-level lexicon.
"""
import json
import os
import subprocess

PUMVR_REPO = "https://github.com/prabhjotschugh/Not-Truly-Multilingual-PuMVR.git"
PUMVR_COMMIT = "2bc54644f578dd847cc5d7378da25bb19090b666"
JSON_REL = os.path.join("Full dataset", "dataset_json.json")


def fetch_pumvr(dest):
    """Shallow-clone at the pinned commit if not already present. Returns the JSON path."""
    path = os.path.join(dest, JSON_REL)
    if os.path.exists(path):
        return path
    subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                    PUMVR_REPO, dest], check=True)
    subprocess.run(["git", "-C", dest, "checkout", PUMVR_COMMIT, "--", JSON_REL],
                   check=True)
    return path


def _clean(t):
    return " ".join((t or "").split())


def load_pumvr_pairs(json_path):
    """
    Returns (sentences, words): lists of dicts with keys gurmukhi, shahmukhi,
    roman, item_id. Items where the three scripts disagree on option count are
    skipped, never silently realigned.
    """
    data = json.load(open(json_path, encoding="utf-8"))
    sents, words, skipped = [], [], 0

    for item in data:
        sc = item.get("scripts", {})
        g, s, r = sc.get("gurmukhi"), sc.get("shahmukhi"), sc.get("roman")
        if not (g and s and r):
            skipped += 1
            continue

        sents.append({"item_id": item["id"],
                      "gurmukhi": _clean(g.get("question")),
                      "shahmukhi": _clean(s.get("question")),
                      "roman": _clean(r.get("question"))})

        og, os_, orr = g.get("options") or [], s.get("options") or [], r.get("options") or []
        if not (len(og) == len(os_) == len(orr)):
            skipped += 1
            continue
        for k, (a, b, c) in enumerate(zip(og, os_, orr)):
            a, b, c = _clean(a), _clean(b), _clean(c)
            if a and b and len(a.split()) == 1 and len(b.split()) == 1:
                words.append({"item_id": f"{item['id']}#o{k}",
                              "gurmukhi": a, "shahmukhi": b, "roman": c})

    return sents, words, skipped
