"""Graded controls and the bracket check, on synthetic draws."""
import os, subprocess, sys, tempfile
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from conditions import collapse_gurmukhi, devowel_gurmukhi

ok = 0
def check(c, msg):
    global ok
    assert c, msg
    ok += 1

t = "ਵਿੱਚ ਕਾਣੀ ਮੇਰੇ ਹੈ ਊਠ ਔਰਤ ਮੂਲ"
l0, l1, l2 = devowel_gurmukhi(t), collapse_gurmukhi(t, 1), collapse_gurmukhi(t, 2)
check("ਣ" in l0 and "ਣ" not in l1, "level 1 merges n")
check(l1.count(" ") == l0.count(" ") == l2.count(" "), "word count kept")
check(not any(ch in l2 for ch in "ੂੌਊਔ"), "level 2 merges waw vowels")
check(l2.split()[2] == "ਮੀਰੇ", "medial e -> i, final e kept")
check(l2.split()[3] == "ਹੇ", "final ai -> e")
check(len(set(l2)) <= len(set(l1)) <= len(set(l0)), "alphabet shrinks")
try:
    collapse_gurmukhi(t, 3); check(False, "bad level accepted")
except ValueError:
    check(True, "")

# Planted: shares 0.9 > 0.8 > 0.7 > 0.6 must read as monotone.
tmp = tempfile.mkdtemp()
rng = np.random.default_rng(0)
L, B = 5, 300
pj = 0.4 + 0.01 * rng.standard_normal((L, B))
arrs = {"punjabi": pj, "point__punjabi": pj.mean(1)}
for s, k in zip(["script_form", "script_form_dv1", "script_form_dv2", "script_form_rt"],
                [0.9, 0.8, 0.7, 0.6]):
    arrs[s] = k * pj; arrs["point__" + s] = k * pj.mean(1)
np.savez(os.path.join(tmp, "bootstrap_draws_flores_m.npz"), **arrs)
p = subprocess.run([sys.executable, os.path.join(ROOT, "tools_bracket_check.py")],
                   env={**os.environ, "MUKHI_RESULTS": tmp}, capture_output=True, text=True)
check(p.returncode == 0, p.stderr)
import json
r = json.load(open(os.path.join(tmp, "bracket_check_flores.json")))["m"]
check(abs(r["shares"]["script_form"]["point"] - 0.9) < 1e-9, "share point")
check(r["monotone_point"] and r["monotone_ci"], "planted monotone detected")
# Break it: swap dv1 above de-voweled.
arrs["script_form_dv1"] = 0.95 * pj; arrs["point__script_form_dv1"] = 0.95 * pj.mean(1)
np.savez(os.path.join(tmp, "bootstrap_draws_flores_m.npz"), **arrs)
subprocess.run([sys.executable, os.path.join(ROOT, "tools_bracket_check.py")],
               env={**os.environ, "MUKHI_RESULTS": tmp}, capture_output=True)
r = json.load(open(os.path.join(tmp, "bracket_check_flores.json")))["m"]
check(not r["monotone_point"], "planted violation detected")
# Per seed: seed 1 violates the ordering, seeds 0 and 2 respect it.
arrs["script_form_dv1"] = 0.8 * pj; arrs["point__script_form_dv1"] = 0.8 * pj.mean(1)
for sd, ks in ((0, [0.9, 0.8, 0.7, 0.6]), (1, [0.9, 0.95, 0.7, 0.6]), (2, [0.9, 0.8, 0.7, 0.6])):
    arrs["seed%d__punjabi" % sd] = pj; arrs["point__seed%d__punjabi" % sd] = pj.mean(1)
    for nm, k in zip(["script_form", "script_form_dv1", "script_form_dv2", "script_form_rt"], ks):
        arrs["seed%d__%s" % (sd, nm)] = k * pj; arrs["point__seed%d__%s" % (sd, nm)] = k * pj.mean(1)
np.savez(os.path.join(tmp, "bootstrap_draws_flores_m.npz"), **arrs)
subprocess.run([sys.executable, os.path.join(ROOT, "tools_bracket_check.py")],
               env={**os.environ, "MUKHI_RESULTS": tmp}, capture_output=True)
r = json.load(open(os.path.join(tmp, "bracket_check_flores.json")))["m"]
check(r["monotone_point"], "seed-mean ordering holds")
check(set(r["per_seed"]) == {"0", "1", "2"}, "all three seeds checked")
check(r["per_seed"]["0"]["monotone_point"] and not r["per_seed"]["1"]["monotone_point"],
      "per-seed violation detected in seed 1 only")
check(not r["monotone_every_seed_point"], "summary flags the failing seed")
print(f"test_bracket: {ok} assertions passed")
