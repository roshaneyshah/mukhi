"""Quantify what de-voweling removes, and what it buys for transliteration.

This is a reproducible pipeline stage. It reads the same condition files the
main experiments read, so its numbers always match the corpus the paper
reports on. An earlier version of these numbers lived in a hand made
results/devowel_check.json with no script behind it; that file predated the
NFC fix (Finding 9) and was computed on PuMVR rather than on FLORES or SIB.

Outputs results/devowel_check.json with provenance recorded.

Usage:
    python tools_devowel_check.py
"""
import json
import os
import sys
import unicodedata
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from conditions import UNWRITTEN_IN_SHAHMUKHI, devowel_gurmukhi  # noqa: E402
from translit_ctx import ctx_shah_to_gur  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _cer(hyp, ref):
    """Levenshtein distance over characters, divided by reference length."""
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(ref) + 1))
    for i, a in enumerate(hyp, 1):
        cur = [i] + [0] * len(ref)
        for j, b in enumerate(ref, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b))
        prev = cur
    return prev[-1] / len(ref)


def _corpus_cer(hyps, refs):
    """Corpus level CER: total edits over total reference characters."""
    num = den = 0
    for h, r in zip(hyps, refs):
        num += _cer(h, r) * len(r)
        den += len(r)
    return num / den if den else 0.0


def removal_fraction(gur_texts):
    """Fraction of Gurmukhi characters the de-voweling rule removes."""
    total = sum(len(t) for t in gur_texts)
    kept = sum(len(devowel_gurmukhi(t)) for t in gur_texts)
    return (total - kept) / total if total else 0.0


def main():
    out = {
        "computed": str(date.today()),
        "source": "results/conditions_flores.json, results/conditions_sib.json",
        "script": "tools_devowel_check.py",
        "marks_removed": sorted(UNWRITTEN_IN_SHAHMUKHI),
        "nfc_applied": True,
    }

    # Removal fraction, on both corpora the paper uses.
    fl = json.load(open(os.path.join(RESULTS, "conditions_flores.json")))
    fl_gur = fl["conditions"]["pan_Guru"]
    out["flores"] = {
        "n_sentences": len(fl_gur),
        "n_chars": sum(len(t) for t in fl_gur),
        "frac_chars_removed": removal_fraction(fl_gur),
    }

    sib = json.load(open(os.path.join(RESULTS, "conditions_sib.json")))
    sib_gur = [t for sp in ("train", "validation", "test") for t in sib[sp]["pan_Guru"]]
    out["sib"] = {
        "n_sentences": len(sib_gur),
        "n_chars": sum(len(t) for t in sib_gur),
        "frac_chars_removed": removal_fraction(sib_gur),
    }

    # Sanity: the rule must remove only the three marks and nothing else.
    bad = 0
    for t in fl_gur[:200]:
        d = devowel_gurmukhi(t)
        if [c for c in t if c not in UNWRITTEN_IN_SHAHMUKHI] != list(d):
            bad += 1
    out["flores_rule_violations_in_first_200"] = bad

    # What de-voweling buys: CER of rule transliterated Shahmukhi against
    # Gurmukhi, versus against de-voweled Gurmukhi, on SIB where the
    # transliteration is already stored.
    hyp = [unicodedata.normalize("NFC", t)
           for sp in ("train", "validation", "test") for t in sib[sp]["pan_Shah_to_Guru"]]
    ref_full = sib_gur
    ref_dev = [devowel_gurmukhi(t) for t in sib_gur]
    out["sib_translit"] = {
        "n_pairs": len(hyp),
        "cer_vs_gurmukhi": _corpus_cer(hyp, ref_full),
        "cer_vs_devowel": _corpus_cer(hyp, ref_dev),
    }

    # Same contrast on FLORES, transliterating here so the two corpora agree
    # on method.
    fl_shah = fl["conditions"]["pan_Shah"]
    fl_hyp = [unicodedata.normalize("NFC", ctx_shah_to_gur(t)) for t in fl_shah]
    out["flores_translit"] = {
        "n_pairs": len(fl_hyp),
        "cer_vs_gurmukhi": _corpus_cer(fl_hyp, fl_gur),
        "cer_vs_devowel": _corpus_cer(fl_hyp, [devowel_gurmukhi(t) for t in fl_gur]),
    }

    path = os.path.join(RESULTS, "devowel_check.json")
    if os.path.exists(path):
        os.replace(path, path + ".pre_reproducible")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("\nwrote", path)


if __name__ == "__main__":
    main()
