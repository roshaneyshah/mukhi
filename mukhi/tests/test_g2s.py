import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from translit_g2s import gur_to_shah, normalise_shah
from translit_eval import levenshtein, split_pairs

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("PASS  " if cond else "FAIL  ") + name + ("   " + detail if detail else ""))

# gold spellings from PuMVR
for g, s_, why in [("ਹੱਥ", "ہتھ", "aspirate + unmarked gemination"),
                   ("ਪੈਰ", "پیر", "ai matra medial -> ye"),
                   ("ਕੰਨ", "کن", "tippi before na is gemination"),
                   ("ਲੰਬਾ", "لمبا", "nasal before labial -> mim"),
                   ("ਹੋਸ਼ਿਆਰ", "ہوشیار", "sihari before independent vowel -> ye"),
                   ("ਆਇਤ", "آیت", "medial i before consonant -> ye"),
                   ("ਮੇਜ਼", "میز", "nukta letter za")]:
    got = gur_to_shah(g)
    check(why, normalise_shah(got) == normalise_shah(s_), f"{g} -> {got} (gold {s_})")

check("final e -> bari ye", gur_to_shah("ਖੱਬੇ").endswith("ے"), gur_to_shah("ਖੱਬੇ"))
check("danda -> urdu full stop", gur_to_shah("।") == "۔")
check("empty", gur_to_shah("") == "")
check("normalise unifies arabic ye/kaf/he", normalise_shah("يكه") == "یکہ")

J = os.environ.get("PUMVR_JSON", os.path.join(os.path.dirname(__file__), "..", "data", "pumvr", "Full dataset", "dataset_json.json"))
if os.path.exists(J):
    from pumvr import load_pumvr_pairs
    S, W, _ = load_pumvr_pairs(J)
    dev, test = split_pairs(W)
    e = sum(levenshtein(normalise_shah(gur_to_shah(g)), normalise_shah(s)) for g, s in test)
    c = sum(len(normalise_shah(s)) for g, s in test)
    check("REGRESSION held-out word CER <= 0.07", e / c <= 0.07, f"{e/c:.4f}")
else:
    print("SKIP  PuMVR not fetched (run --stage translit_eval first)")

print()
nf = sum(1 for _, ok, _ in results if not ok)
print(f"{len(results)-nf}/{len(results)} passed")
sys.exit(1 if nf else 0)
