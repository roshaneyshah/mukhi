#!/usr/bin/env python3
"""Merge the FINDING_09 repair run into the full results.

The repair run recomputed only the two contrasts the normalisation fix
touches. Everything else in each model's file was measured on conditions the
fix left byte-identical, so it is kept. After substituting, Benjamini-Hochberg
is recomputed over the whole family (every layer x contrast x statistic of
that model), because the family is what BH corrects over and two of its
members changed.

Usage:  python tools_merge_nfc.py <dir-with-_nfc-files>  [--dry-run]
"""
import json, os, shutil, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from analysis import apply_bh, decompose_punjabi_gap, crossover_layer

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
REPAIRED = ("pan_Guru|pan_Shah", "pan_Guru|pan_Guru_devowel")


def merge_one(full_path, nfc_path, dry=False):
    full = json.load(open(full_path, encoding="utf-8"))
    nfc = json.load(open(nfc_path, encoding="utf-8"))

    if "contrasts" not in nfc:
        return f"SKIP {os.path.basename(nfc_path)}: no contrasts block"
    missing = [c for c in REPAIRED if c not in nfc["contrasts"]]
    if missing:
        return f"SKIP {os.path.basename(nfc_path)}: repair run lacks {missing}"
    for key in ("model", "kind", "n", "random_init", "seed"):
        if full.get(key) != nfc.get(key):
            return (f"SKIP {os.path.basename(nfc_path)}: {key} differs "
                    f"({full.get(key)!r} vs {nfc.get(key)!r})")

    nl_full = len(full["contrasts"][REPAIRED[0]]["paired"])
    nl_nfc = len(nfc["contrasts"][REPAIRED[0]]["paired"])
    if nl_full != nl_nfc:
        return f"SKIP {os.path.basename(nfc_path)}: {nl_full} layers vs {nl_nfc}"

    before = {c: full["contrasts"][c]["paired"][-1]["delta"] for c in REPAIRED}
    for c in REPAIRED:
        full["contrasts"][c] = nfc["contrasts"][c]
        full.setdefault("repaired_contrasts", [])
        if c not in full["repaired_contrasts"]:
            full["repaired_contrasts"].append(c)
    full["extraction"] = {**full.get("extraction", {}),
                          **{k: v for k, v in nfc.get("extraction", {}).items()}}
    full["normalisation"] = "NFC at source (FINDING_09)"

    full["bh_family_size"] = apply_bh(full["contrasts"])
    dec = decompose_punjabi_gap(full["contrasts"])
    if dec:
        full["decomposition"] = dec
    res = full["contrasts"]
    if "pan_Guru|pan_Shah" in res and "pan_Shah|urd_Arab" in res:
        full["crossover_layer"] = crossover_layer(
            script_sel=[r["selectivity"] for r in res["pan_Guru|pan_Shah"]["probe"]],
            language_sel=[r["selectivity"] for r in res["pan_Shah|urd_Arab"]["probe"]])

    after = {c: full["contrasts"][c]["paired"][-1]["delta"] for c in REPAIRED}
    delta = ", ".join("%s %.3f -> %.3f" % (c.split("|")[1], before[c], after[c])
                      for c in REPAIRED)
    if not dry:
        shutil.copy(full_path, full_path + ".pre_nfc")
        json.dump(full, open(full_path, "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)
    return "ok   %-42s  %s  (BH family %d)" % (
        os.path.basename(full_path), delta, full["bh_family_size"])


def main():
    src = sys.argv[1]
    dry = "--dry-run" in sys.argv
    found = sorted(f for f in os.listdir(src)
                   if f.startswith("main_flores_") and f.endswith("_nfc.json")
                   and not f.startswith("main_flores_summary"))
    if not found:
        sys.exit(f"no *_nfc.json files in {src}")
    for f in found:
        full_name = f.replace("_nfc.json", ".json")
        full_path = os.path.join(RESULTS, full_name)
        if not os.path.exists(full_path):
            print(f"SKIP {f}: no {full_name} to merge into")
            continue
        print(merge_one(full_path, os.path.join(src, f), dry=dry))
    print("\nDRY RUN, nothing written" if dry else
          "\nmerged; originals kept as *.pre_nfc")


if __name__ == "__main__":
    main()
