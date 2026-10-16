#!/usr/bin/env python3
"""Every condition must be NFC, and no two conditions may differ by
normalisation alone (FINDING_09)."""
import sys, os, unicodedata
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from conditions import build_conditions, devowel_gurmukhi, _nfc

ok = fail = 0
def check(cond, msg):
    global ok, fail
    if cond: ok += 1
    else:
        fail += 1
        print("FAIL:", msg)

# the five nukta letters that XLM-R maps to UNK, precomposed
PRECOMPOSED = "ਸ਼ਜ਼ਫ਼ਖ਼ਗ਼"
NUKTA = "਼"

for ch in PRECOMPOSED:
    n = _nfc(ch)
    check(len(n) == 2 and n[1] == NUKTA,
          f"NFC should decompose U+{ord(ch):04X} to base + nukta, got {n!r}")

ids = ["a", "b"]
flores = {"pan_Guru": ["ਸ਼ਹਿਰ ਜ਼ਿਲ੍ਹਾ ਫ਼ੌਜ ਖ਼ਬਰ ਗ਼ਲਤੀ।", "ਪੰਜਾਬੀ ਬੋਲੀ ਹੈ।"],
          "urd_Arab": ["شہر۔", "پنجابی۔"],
          "hin_Deva": ["शहर।", "पंजाबी।"]}
c = build_conditions(ids, flores, rom=False)

for name, texts in c.items():
    for t in texts:
        check(unicodedata.normalize("NFC", t) == t, f"{name} is not NFC: {t!r}")
        check(not any(ch in PRECOMPOSED for ch in t),
              f"{name} still holds a precomposed nukta letter: {t!r}")

# de-voweling must not change which letters are present, only the three marks
REMOVED = set("ਿੁੱ")
for g, dv in zip(c["pan_Guru"], c["pan_Guru_devowel"]):
    check("".join(ch for ch in g if ch not in REMOVED) == dv,
          f"de-voweling changed more than the three marks: {g!r} -> {dv!r}")
    check(g.count(NUKTA) == dv.count(NUKTA),
          f"de-voweling changed the nukta count: {g!r} -> {dv!r}")

# the historical bug: devowel normalised, the source did not
raw = "ਸ਼ਹਿਰ"          # precomposed sha + sihari
check(devowel_gurmukhi(raw).count(NUKTA) == 1,
      "devowel should preserve the nukta as a combining mark")
check(unicodedata.normalize("NFC", raw) != raw,
      "test fixture should be a non-NFC string")

# --- every entry point that builds conditions must normalise at the source ---
# FLORES was fixed first; SIB and PuMVR are separate code paths in run_all.py
# and were missed, which left the intervention arm carrying the confound.
import re
ra = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "run_all.py"), encoding="utf-8").read()

sib = ra[ra.index('load_dataset("Davlan/sib200"'):]
sib = sib[:sib.index("save(\"conditions_sib.json\"")]
check("_nfc(r[\"text\"])" in sib,
      "the SIB path must NFC-normalise its Gurmukhi at the source")
check("devowel_gurmukhi(t) for t in texts" in sib,
      "the SIB de-voweled control must derive from the normalised texts")

pum = ra[ra.index("def _pumvr_conditions"):]
pum = pum[:pum.index("PUMVR_CONTRASTS")]
check('_nfc(x["gurmukhi"])' in pum and '_nfc(x["shahmukhi"])' in pum,
      "the PuMVR path must NFC-normalise its Gurmukhi and Shahmukhi")

for path, label in ((ra, "run_all.py"),):
    for raw in ('"pan_Guru": texts,', '"pan_Guru": g,'):
        if raw in path:
            i = path.index(raw)
            window = path[max(0, i - 400):i]
            check("_nfc" in window,
                  f"{label}: {raw} is assigned without a visible _nfc call nearby")

# --- gated loads must pass the token explicitly ---
# A run that authenticated one day failed the next on a refreshed Kaggle image
# because the libraries stopped picking HF_TOKEN up from the environment.
import conditions as _c
src_cond = open(_c.__file__, encoding="utf-8").read()
flores = src_cond[src_cond.index("def load_flores"):]
flores = flores[:flores.index("def build_conditions")]
check("token=token" in flores,
      "load_flores must pass token= to load_dataset")
check('os.environ.get("HF_TOKEN")' in flores
      and 'os.environ.get("HUGGING_FACE_HUB_TOKEN")' in flores,
      "load_flores must read both token environment variables")

check("token=_hf_token()" in ra,
      "the SIB load must pass token=_hf_token()")
check("def _hf_token()" in ra, "run_all.py must define _hf_token")

print(f"{ok}/{ok + fail} passed")
sys.exit(1 if fail else 0)
