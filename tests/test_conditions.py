import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from conditions import serbian_cyr_to_lat, devowel_gurmukhi, build_conditions, CONTRASTS, romanize

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("PASS  " if cond else "FAIL  ") + name + ("   " + detail if detail else ""))

# Serbian: known correspondences
cases = [("Добар дан", "Dobar dan"), ("Љубав", "Ljubav"), ("ЉУБАВ", "LJUBAV"),
         ("Његош", "Njegoš"), ("џеп", "džep"), ("Ђорђе", "Đorđe"), ("ћуприја", "ćuprija"),
         ("Београд, 2026.", "Beograd, 2026."), ("жути чај", "žuti čaj")]
for cyr, lat in cases:
    got = serbian_cyr_to_lat(cyr)
    check(f"serbian {cyr}", got == lat, f"-> {got}")

# Serbian mapping is lossless: 30 letters, injective
from conditions import _SR
check("serbian map has all 30 letters", len(_SR) == 30, str(len(_SR)))
check("serbian map is injective", len(set(_SR.values())) == 30)

# de-voweling
check("devowel removes sihari/aunkar/addak", devowel_gurmukhi("ਟਿੱਕਾ ਕੁਰਸੀ") == "ਟਕਾ ਕਰਸੀ",
      devowel_gurmukhi("ਟਿੱਕਾ ਕੁਰਸੀ"))
check("devowel keeps long vowels", devowel_gurmukhi("ਪੰਜਾਬੀ") == "ਪੰਜਾਬੀ")
check("devowel empty", devowel_gurmukhi("") == "")

# romanisation
r = romanize(["ਪੰਜਾਬੀ", "پنجابی"], "pan")
check("romanize returns latin", all(x.isascii() for x in r), str(r))

# build_conditions with stub data
ids = ["a", "b"]
fl = {"pan_Guru": ["ਪੰਜਾਬੀ ਭਾਸ਼ਾ", "ਟਿੱਕਾ"], "urd_Arab": ["اردو", "زبان"],
      "hin_Deva": ["हिन्दी", "भाषा"], "srp_Cyrl": ["Добар", "дан"]}
c = build_conditions(ids, fl, gur_to_shah=lambda t: "پنجابی", rom=True)
check("all conditions built", set(c) >= {a for a, b, *_ in CONTRASTS} | {b for a, b, *_ in CONTRASTS},
      str(sorted(c)))
check("all conditions aligned", all(len(v) == 2 for v in c.values()))
try:
    build_conditions(ids, fl, gur_to_shah=lambda t: "x", rom=False)
    ok = True
except Exception:
    ok = False
check("rom=False works", ok)

print()
nf = sum(1 for _, ok, _ in results if not ok)
print(f"{len(results)-nf}/{len(results)} passed")
sys.exit(1 if nf else 0)
