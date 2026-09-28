import sys, os, tempfile, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from data import (filter_pairs, stratified_sample, prioritise_for_audit,
                  write_audit_sheet, score_audit, corpus_error_estimate,
                  _wilson, AUDIT_COLUMNS)

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("PASS  " if cond else "FAIL  ") + name + ("   " + detail if detail else ""))

G = "ਪੰਜਾਬ"   # gurmukhi word
S = "پنجاب"   # shahmukhi word

def gpair(nw_g=6, nw_s=6, tag=""):
    return {"gurmukhi": " ".join([G+tag]*nw_g), "shahmukhi": " ".join([S+tag]*nw_s)}

# --- filtering --------------------------------------------------------------
pairs = [gpair(6,6,"a"), gpair(2,2,"b"), gpair(60,60,"c"),
         {"gurmukhi":"", "shahmukhi":S}, gpair(6,6,"a"),
         {"gurmukhi": " ".join([S]*6), "shahmukhi": " ".join([S]*6)}]
kept, reasons = filter_pairs(pairs)
check("filter keeps the one good unique pair", len(kept)==1, f"kept={len(kept)} {reasons}")
check("filter counts short", reasons["short"]==1, str(reasons))
check("filter counts long", reasons["long"]==1, str(reasons))
check("filter counts empty", reasons["empty"]==1, str(reasons))
check("filter counts dup", reasons["dup"]==1, str(reasons))
check("filter rejects wrong-script gurmukhi field", reasons["script"]==1, str(reasons))
check("filter assigns pair_id", "pair_id" in kept[0])

# --- stratified sampling ----------------------------------------------------
many = []
for i in range(400):
    nw = 4 + (i % 20)
    many.append(gpair(nw, nw, f"x{i}"))
kept2, _ = filter_pairs(many)
samp = stratified_sample(kept2, n=100, seed=0)
check("stratified sample returns exactly n", len(samp)==100, str(len(samp)))
check("stratified sample has unique ids", len({p["pair_id"] for p in samp})==100)
lens = [p["n_words_gur"] for p in samp]
check("stratified sample spans length range", (max(lens)-min(lens))>10,
      f"{min(lens)}-{max(lens)}")
check("sample smaller than n returns all", len(stratified_sample(kept2[:20], n=100))==20)

# --- audit prioritisation ---------------------------------------------------
# checker returns the corpus Gurmukhi for 2/3 of pairs (clean) and garbage for
# 1/3 (suspicious), so we know which stratum each pair should land in.
gold = {p["shahmukhi"]: p["gurmukhi"] for p in kept2}
def checker(s):
    return gold[s] if hash(s) % 3 else "\u0a38" * 40
rows, stats = prioritise_for_audit(kept2, checker, n=60, flag_frac=0.5, seed=0)
check("audit returns requested n", len(rows)==60, str(len(rows)))
n_flag = sum(1 for r in rows if r["audit_stratum"]=="flagged")
check("audit honours flag_frac", n_flag==30, f"got {n_flag}")
check("audit reports corpus flag rate", 0.0 < stats["flag_rate"] < 1.0, f"{stats['flag_rate']:.3f}")
check("flagged rows really are high-suspicion",
      all(r["suspicion"] > stats["tau"] for r in rows if r["audit_stratum"]=="flagged"))
check("unflagged rows really are low-suspicion",
      all(r["suspicion"] <= stats["tau"] for r in rows if r["audit_stratum"]=="unflagged"))
check("helper column empty when no helper given", all(r["sys_a"]=="" for r in rows))
rows_h, _ = prioritise_for_audit(kept2, checker, n=10, helper=lambda s: "H", seed=0)
check("helper column filled when helper given", all(r["sys_a"]=="H" for r in rows_h))

# --- sheet round-trip -------------------------------------------------------
tmp = tempfile.mkdtemp()
path = os.path.join(tmp, "audit.csv")
write_audit_sheet(rows, path)
back = list(csv.DictReader(open(path, encoding="utf-8")))
check("sheet round-trips row count", len(back)==60, str(len(back)))
check("sheet has all columns", list(back[0].keys())==AUDIT_COLUMNS)

# --- scoring ----------------------------------------------------------------
for i, r in enumerate(back):
    if r["audit_stratum"]=="flagged":
        r["verdict"] = "error" if i % 2 == 0 else "ok"
        r["error_type"] = "misalignment" if i % 2 == 0 else "ok"
    else:
        r["verdict"] = "error" if i % 10 == 0 else "ok"
        r["error_type"] = "vowel" if i % 10 == 0 else "ok"
with open(path,"w",newline="",encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=AUDIT_COLUMNS); w.writeheader(); w.writerows(back)

sc = score_audit(path)
check("scoring reads both strata", set(sc["strata"])=={"flagged","unflagged"})
check("scoring counts judged rows", sc["strata"]["flagged"]["n_judged"]==30,
      str(sc["strata"]["flagged"]["n_judged"]))
check("flagged error rate higher than unflagged",
      sc["strata"]["flagged"]["error_rate"] > sc["strata"]["unflagged"]["error_rate"])
check("error-type histogram populated", "misalignment" in sc["error_types"])
lo, hi = sc["strata"]["flagged"]["ci95"]
check("CI brackets the point estimate", lo <= sc["strata"]["flagged"]["error_rate"] <= hi)

# --- corpus reweighting -----------------------------------------------------
est = corpus_error_estimate(sc, stats["flag_rate"])
ef, eu = sc["strata"]["flagged"]["error_rate"], sc["strata"]["unflagged"]["error_rate"]
check("corpus estimate lies between the two strata rates", min(ef,eu) <= est <= max(ef,eu),
      f"est={est:.4f}")
check("corpus estimate equals the weighted formula",
      abs(est - (stats["flag_rate"]*ef + (1-stats["flag_rate"])*eu)) < 1e-12)
check("corpus estimate below flagged-stratum rate (no overstatement)", est < ef)

# --- Wilson interval sanity -------------------------------------------------
lo0, hi0 = _wilson(0, 30)
check("Wilson handles 0 errors without collapsing to [0,0]", hi0 > 0.05, f"[{lo0:.3f},{hi0:.3f}]")
lo1, hi1 = _wilson(15, 30)
check("Wilson centred near 0.5 at k=n/2", abs((lo1+hi1)/2 - 0.5) < 0.02)
check("Wilson narrows with n", (_wilson(50,1000)[1]-_wilson(50,1000)[0]) <
      (_wilson(5,100)[1]-_wilson(5,100)[0]))

print()
nf = sum(1 for _,ok,_ in results if not ok)
print(f"{len(results)-nf}/{len(results)} passed")
sys.exit(1 if nf else 0)
