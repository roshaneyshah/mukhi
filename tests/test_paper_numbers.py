"""Assert the draft's prose agrees with results/paper_numbers.json.

Three stale numbers survived into the draft because nothing tied the prose
to the result files: §3.3's de-voweling fraction came from a hand made JSON
with no script behind it, and §5.2's displacement figures predated the NFC
repair. This test closes that gap. If a run changes the numbers, it fails
until either the draft or the expectation is updated deliberately.

Run tools_paper_numbers.py and tools_devowel_check.py first.
"""
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
DRAFT = os.path.join(ROOT, "paper", "draft.md")

n_fail = 0


def check(label, expected, text=None):
    """Assert `expected` appears verbatim in the draft."""
    global n_fail
    body = text if text is not None else DRAFT_TEXT
    ok = expected in body
    if not ok:
        n_fail += 1
    print("%-4s %-46s %s" % ("ok" if ok else "FAIL", label, expected))


if not os.path.exists(DRAFT):
    print("no draft at %s, nothing to check" % DRAFT)
    sys.exit(0)

for needed in ("paper_numbers.json", "devowel_check.json"):
    if not os.path.exists(os.path.join(RESULTS, needed)):
        print("missing results/%s; run its tools_*.py first" % needed)
        sys.exit(1)

def _flat(t):
    """Collapse whitespace runs. The draft is hard-wrapped at 78 columns, so a
    quoted interval like "[0.320, 0.326]" can be split across a line break and
    would otherwise read as missing when it is present and correct."""
    return " ".join(t.split())


DRAFT_TEXT = _flat(open(DRAFT).read())
README = os.path.join(ROOT, "README.md")
README_TEXT = _flat(open(README).read()) if os.path.exists(README) else ""
PN = json.load(open(os.path.join(RESULTS, "paper_numbers.json")))
DV = json.load(open(os.path.join(RESULTS, "devowel_check.json")))

M = PN["models"]


def f3(x):
    return "%.3f" % x


def f1(x):
    return "%.1f" % x


# --- 5.2 and 5.7: displacement curves, trained minus random ---
# XLM-R large's published main run had one random-init seed. The draft quotes
# its three-seed values, checked against the bootstrap files further down.
for model, label in (("xlm-roberta-base", "XLM-R base"),
                     ("muril-base-cased", "MuRIL")):
    pj = M[model]["punjabi"]
    check("%s Punjabi embedding" % label, f3(pj["embedding"]))
    check("%s Punjabi trough" % label, f3(pj["trough"]))
    check("%s Punjabi final" % label, f3(pj["final"]))

for model, label in (("xlm-roberta-base", "XLM-R base"),
                     ("xlm-roberta-large", "XLM-R large"),
                     ("muril-base-cased", "MuRIL")):
    check("%s Serbian trough" % label, f3(M[model]["serbian"]["trough"]))

# The no-script control (romanised Shahmukhi against romanised Urdu). Both
# of the two numbers missed in the first reconciliation pass sat here and in
# MuRIL's Serbian trough, and both were quoted on the sign-flip-subtracted
# "delta" basis rather than the "consistency" basis the paper uses
# everywhere else, so assert them explicitly.
for model, label in (("xlm-roberta-base", "XLM-R base"),
                     ("muril-base-cased", "MuRIL")):
    ns = M[model]["no_script"]
    check("%s no-script embedding" % label, f3(ns["embedding"]))
    check("%s no-script trough" % label, f3(ns["trough"]))
    check("%s no-script final" % label, f3(ns["final"]))

check("XLM-R base Serbian embedding", f3(M["xlm-roberta-base"]["serbian"]["embedding"]))
check("XLM-R base Serbian final", f3(M["xlm-roberta-base"]["serbian"]["final"]))

# The trough layers are quoted as "at layer N".
check("XLM-R base trough layer", "layer %d" % M["xlm-roberta-base"]["punjabi"]["trough_layer"])
check("XLM-R large trough layer", "layer %d" % M["xlm-roberta-large"]["punjabi"]["trough_layer"])

# --- 5.3: information contrast and the script-form ratio ---
inf = M["xlm-roberta-base"]["information"]
check("XLM-R base information embedding", f3(inf["embedding"]))
check("XLM-R base information final", f3(inf["final"]))

for model, label in (("xlm-roberta-base", "XLM-R base"),
                     ("muril-base-cased", "MuRIL")):
    r = M[model]["script_form_ratio"]
    check("%s form ratio mean" % label, f3(r["mean"]))
    check("%s form ratio min" % label, f3(r["min"]))
    check("%s form ratio max" % label, f3(r["max"]))
    check("%s raw form ratio" % label, f3(M[model]["script_form_ratio_raw"]["mean"]))
    check("%s script-form trough" % label, f3(M[model]["script_form"]["trough"]))

# The abstract states the ratio range exactly, so it must be the real
# min and max of the three per-model means.
means = sorted(M[m]["script_form_ratio"]["mean"] for m in M if "script_form_ratio" in M[m])
check("ratio range high", f3(means[-1]))
check("XLM-R large raw form ratio", f3(M["xlm-roberta-large"]["script_form_ratio_raw"]["mean"]))

# --- 5.5: tokeniser premium ---
fert = PN["fertility"]
check("XLM-R tokeniser premium", "%s%%" % f1(fert["xlm-roberta-base"]["premium_pct"]))
check("MuRIL tokeniser premium", "%s%%" % f1(fert["google/muril-base-cased"]["premium_pct"]))
check("MuRIL Gurmukhi tokens per char", f3(fert["google/muril-base-cased"]["guru_tokens_per_char"]))
check("MuRIL Shahmukhi tokens per char", f3(fert["google/muril-base-cased"]["shah_tokens_per_char"]))

# --- 5.6: intervention ---
for model, label in (("xlm-roberta-base", "XLM-R"), ("muril-base-cased", "MuRIL")):
    iv = PN["intervention"][model]
    for cond in ("pan_Guru", "pan_Shah", "pan_Shah_to_Guru", "pan_Guru_devowel"):
        check("%s probe %s" % (label, cond), f3(iv["acc"][cond]))
    check("%s probe gap" % label, f3(iv["gap"]))
    check("%s probe gain" % label, f3(iv["gain"]))
    check("%s probe gain CI low" % label, f3(iv["gain_ci95"][0]))
    check("%s probe gain CI high" % label, f3(iv["gain_ci95"][1]))
    check("%s gap closed pct" % label, "%d%%" % round(iv["gap_closed_frac"] * 100))

# --- 3.3: de-voweling ---
check("FLORES removal fraction", "%.1f%%" % (DV["flores"]["frac_chars_removed"] * 100))
check("SIB removal fraction", "%.1f%%" % (DV["sib"]["frac_chars_removed"] * 100))
check("FLORES CER vs Gurmukhi", "%.1f%%" % (DV["flores_translit"]["cer_vs_gurmukhi"] * 100))
check("FLORES CER vs de-voweled", "%.1f%%" % (DV["flores_translit"]["cer_vs_devowel"] * 100))
check("SIB CER vs Gurmukhi", "%.1f%%" % (DV["sib_translit"]["cer_vs_gurmukhi"] * 100))
check("SIB CER vs de-voweled", "%.1f%%" % (DV["sib_translit"]["cer_vs_devowel"] * 100))

# --- seed spreads quoted in 5.2 ---
check("XLM-R base seed spread at trough", f3(M["xlm-roberta-base"]["punjabi"]["trough_seed_spread"]))
check("MuRIL seed spread at trough", f3(M["muril-base-cased"]["punjabi"]["trough_seed_spread"]))

# --- README results table: the same curves, quoted again ---
if README_TEXT:
    print()
    for cname in ("punjabi", "serbian"):
        c = M["xlm-roberta-base"][cname]
        row = "| %s | %s (L%d) | %s |" % (f3(c["embedding"]), f3(c["trough"]),
                                          c["trough_layer"], f3(c["final"]))
        check("README XLM-R base %s row" % cname, row, README_TEXT)
    # Large and XL rows come from the three-seed bootstrap runs.
    for path, tag in ((os.path.join(RESULTS, "round3", "bootstrap_flores_round3_xlm-roberta-large.json"), "xlm-roberta-large"),
                      (os.path.join(RESULTS, "round4", "bootstrap_flores_round4_xlm-roberta-xl.json"), "xlm-roberta-xl")):
        cc = json.load(open(path))["per_model"][tag]["contrasts"]
        for cname in ("punjabi", "serbian"):
            c = cc[cname]
            row = "| %s | %s (L%d) | %s |" % (f3(c["embedding"]["point"]), f3(c["trough"]["point"]),
                                              c["trough"]["layer"], f3(c["final"]["point"]))
            check("README %s %s row" % (tag, cname), row, README_TEXT)
    ns = M["xlm-roberta-base"]["no_script"]
    check("README no-script control row",
          "| %s | %s (L%d) | %s |" % (f3(ns["embedding"]), f3(ns["trough"]),
                                      ns["trough_layer"], f3(ns["final"])),
          README_TEXT)

# --- README repository statistics ---
if README_TEXT:
    import glob as _glob
    n_suites = len(_glob.glob(os.path.join(ROOT, "tests", "test_*.py")))
    check("README suite count in status line", "%d test suites" % n_suites, README_TEXT)
    check("README suite count in tree", "%d suites" % n_suites, README_TEXT)

# --- bootstrap intervals (Sections 4.6, 5.2, 5.3, 5.4, 5.7) -------------
# The bootstrap numbers are quoted in five places in the draft. Assert each
# against the saved result files, the same way every other figure is checked.
BOOT = {}
for _m in ("xlm-roberta-base", "xlm-roberta-large", "muril-base-cased"):
    _p = os.path.join(RESULTS, "bootstrap_flores_%s.json" % _m)
    if os.path.exists(_p):
        BOOT[_m] = json.load(open(_p))["per_model"][_m]
_xm = os.path.join(RESULTS, "bootstrap_crossmodel_flores.json")
XM = json.load(open(_xm)) if os.path.exists(_xm) else None

if BOOT:
    print()

    def ci3(d):
        return "[%.3f, %.3f]" % (d["ci95"][0], d["ci95"][1])

    def sci3(d):
        """Signed, as the draft writes differences."""
        return "[%+.3f, %+.3f]" % (d["ci95"][0], d["ci95"][1])

    for model, label in (("xlm-roberta-base", "XLM-R base"),
                         ("xlm-roberta-large", "XLM-R large"),
                         ("muril-base-cased", "MuRIL")):
        if model not in BOOT:
            continue
        m = BOOT[model]
        check("%s trough CI" % label, ci3(m["contrasts"]["punjabi"]["trough"]))
        if "script_form_ratio" in m:
            _r = m["script_form_ratio"]
            check("%s ratio CI" % label,
                  ci3(_r.get("layer_mean") or _r["per_layer_mean"]))
        t = m.get("punjabi_minus_serbian_trough")
        if t:
            check("%s ordering point" % label, "%+.3f" % t["point"])
            check("%s ordering CI" % label, sci3(t))

    # Every per-layer test outside the no-script control must be significant,
    # which is what the draft's "all 204" claim asserts.
    n_real = n_sig = 0
    for m in BOOT.values():
        for lab, c in m["contrasts"].items():
            if lab == "no_script":
                continue
            for r in c["per_layer"]:
                n_real += 1
                n_sig += bool(r.get("significant_bh"))
    allsig = (n_real == n_sig)
    if not allsig:
        n_fail += 1
    print("%-4s %-46s %d of %d" % ("ok" if allsig else "FAIL",
                                   "all non-control per-layer tests significant",
                                   n_sig, n_real))
    check("the draft's count of those tests", "%d per-layer tests" % n_real)

    if XM:
        a = XM["across_models"]
        for lab, name in (("punjabi", "Punjabi"), ("serbian", "Serbian"),
                          ("script_form", "script-form")):
            k = "%s_trough_xlm-roberta-large_minus_xlm-roberta-base" % lab
            if k in a:
                check("scale %s point" % name, "%+.3f" % a[k]["point"])
                if lab != "script_form":
                    check("scale %s CI" % name, sci3(a[k]))
        k = "punjabi_trough_xlm-roberta-large_minus_xlm-roberta-base"
        if k in a:
            check("scale Punjabi p value", "p = %.2f" % a[k]["p_boot"])


# Section 5.3, Table 4: the graded controls, from the bracket run.
_bp = os.path.join(RESULTS, "bracket_check_flores.json")
if os.path.exists(_bp):
    BR = json.load(open(_bp))
    for _m, _lab in (("xlm-roberta-base", "XLM-R base"), ("xlm-roberta-large", "XLM-R large"),
                     ("muril-base-cased", "MuRIL")):
        r = BR[_m]
        ok_ = r["monotone_point"] and r["monotone_ci"]
        if not ok_:
            n_fail += 1
        print("%-4s %-46s %s" % ("ok" if ok_ else "FAIL", "%s share falls at every step" % _lab, ok_))
        for k in ("script_form", "script_form_dv1", "script_form_dv2", "script_form_rt"):
            v = r["shares"][k]
            check("%s %s share and CI" % (_lab, k), "%.3f [%.3f, %.3f]" % (v["point"], v["ci95"][0], v["ci95"][1]))
    st = BR["xlm-roberta-base"]["steps"]["script_form_minus_script_form_dv1"]
    check("smallest bracket step", "%+.3f [%+.3f, %+.3f]" % (st["point"], st["ci95"][0], st["ci95"][1]))
else:
    n_fail += 1
    print("FAIL bracket_check_flores.json missing")

# Section 5.8: the mean-pooled decoder.
_dp = os.path.join(RESULTS, "decoder", "bootstrap_draws_flores_Qwen2.5-1.5B_poolmean.npz")
_dj = os.path.join(RESULTS, "bootstrap_flores_poolmean_decoder_meanpool.json")
if os.path.exists(_dp) and os.path.exists(_dj):
    import numpy as _np
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from bootstrap import summarise as _summ
    D = _np.load(_dp)
    P, S, N = D["point__punjabi"], D["point__serbian"], D["point__no_script"]
    L = len(P) // 2
    check("decoder Punjabi embedding", f3(P[0]))
    check("decoder Punjabi middle layer", f3(P[L]))
    check("decoder Punjabi last", f3(P[-1]))
    check("decoder Serbian layer 14", f3(S[L]))
    _d = _summ(D["punjabi"][L] - D["serbian"][L], point=float(P[L] - S[L]))
    check("decoder Punjabi minus Serbian", "%+.3f [%+.3f, %+.3f]" % (_d["point"], _d["ci95"][0], _d["ci95"][1]))
    _all = bool((S > P).all())
    if not _all:
        n_fail += 1
    print("%-4s %-46s %s" % ("ok" if _all else "FAIL", "decoder Serbian above Punjabi at every layer", _all))
    check("decoder no-script max", "%.2f" % N.max())
    _sr = json.load(open(_dj))["per_model"]["Qwen2.5-1.5B_poolmean"]["script_form_ratio"]["layer_mean"]
    check("decoder share", "%.3f [%.3f, %.3f]" % (_sr["point"], _sr["ci95"][0], _sr["ci95"][1]))
else:
    n_fail += 1
    print("FAIL decoder results missing")


# --- Rounds 3 and 4: three-seed encoder runs, XL, PuMVR, decoders, resource ---
import numpy as _np2
R3, R4 = os.path.join(RESULTS, "round3"), os.path.join(RESULTS, "round4")


def _bj(path, tag):
    return json.load(open(path))["per_model"][tag]


def _ci(v, fmt="%+.3f"):
    return ("%s [%s, %s]" % (fmt, fmt, fmt)) % (v["point"], v["ci95"][0], v["ci95"][1])


try:
    L3 = _bj(os.path.join(R3, "bootstrap_flores_round3_xlm-roberta-large.json"), "xlm-roberta-large")["contrasts"]
    XL = _bj(os.path.join(R4, "bootstrap_flores_round4_xlm-roberta-xl.json"), "xlm-roberta-xl")["contrasts"]
    check("large Punjabi trough", f3(L3["punjabi"]["trough"]["point"]))
    check("large Serbian trough", f3(L3["serbian"]["trough"]["point"]))
    check("large de-voweled trough", f3(L3["script_form"]["trough"]["point"]))
    check("XL Punjabi trough", f3(XL["punjabi"]["trough"]["point"]))
    check("XL Serbian trough", "%.3f" % XL["serbian"]["trough"]["point"])
    check("XL de-voweled trough", f3(XL["script_form"]["trough"]["point"]))
    check("XL Hindi-Urdu trough", "%.3f" % XL["hindi_urdu"]["trough"]["point"])
    for m, v in (("xlm-roberta-base", "0.094"), ("xlm-roberta-large", "0.090")):
        c = _bj(os.path.join(R3, "bootstrap_flores_round3_%s.json" % m), m)["contrasts"]
        check("%s Hindi-Urdu trough" % m, f3(c["hindi_urdu"]["trough"]["point"]))

    BX = json.load(open(os.path.join(R4, "bracket_check_flores_with_xl.json")))
    lab = {"xlm-roberta-base": "base", "xlm-roberta-large": "large",
           "xlm-roberta-xl": "XL", "muril-base-cased": "MuRIL"}
    for m, r in BX.items():
        ok_ = r["monotone_point"] and r["monotone_ci"]
        if not ok_:
            n_fail += 1
        print("%-4s %-46s %s" % ("ok" if ok_ else "FAIL", "%s share falls at every step" % lab[m], ok_))
        for k, v in r["shares"].items():
            check("%s %s share" % (lab[m], k), "%.3f [%.3f, %.3f]" % (v["point"], v["ci95"][0], v["ci95"][1]))
        ends = all(v["shares"]["script_form"]["point"] > v["shares"]["script_form_rt"]["point"]
                   for v in r["per_seed"].values())
        if not ends:
            n_fail += 1
        print("%-4s %-46s %s" % ("ok" if ends else "FAIL", "%s ends ordered in every seed" % lab[m], ends))
    lo = min(v["shares"]["script_form_rt"]["point"] for r in BX.values() for v in r["per_seed"].values())
    hi = max(v["shares"]["script_form"]["point"] for r in BX.values() for v in r["per_seed"].values())
    check("per-seed ends span", "span %.2f to %.2f" % (lo, hi))
    s2 = BX["xlm-roberta-base"]["per_seed"]["2"]["shares"]
    check("base seed 2 reversal", "(%.3f to %.3f)" % (s2["script_form"]["point"], s2["script_form_dv1"]["point"]))
    fails = [m for m, r in BX.items() if not r["monotone_every_seed_point"]]
    _f = fails == ["xlm-roberta-base"]
    if not _f:
        n_fail += 1
    print("%-4s %-46s %s" % ("ok" if _f else "FAIL", "only XLM-R base has a seed with a reversed step", fails))

    XM2 = json.load(open(os.path.join(R4, "bootstrap_crossmodel_flores_with_xl.json")))
    px = XM2["per_model"]["xlm-roberta-xl"]
    for pre in ("", "shared__", "rms__", "ratio__"):
        check("XL Table 6 %s" % (pre or "consistency"), _ci(px[pre + "punjabi_minus_serbian_trough"]))
    a = XM2["across_models"]
    K = "%s_trough_xlm-roberta-xl_minus_xlm-roberta-large"
    check("XL-large Punjabi", _ci(a[K % "punjabi"]))
    check("XL-large Serbian", _ci(a[K % "serbian"]))
    check("XL-large ratio Punjabi", "%+.3f" % a[K % "ratio__punjabi"]["point"])
    check("XL-large ratio Serbian", "%+.3f" % a[K % "ratio__serbian"]["point"])
    check("XL-large shared", "+%.3f and %.3f" % (a[K % "shared__punjabi"]["point"], a[K % "shared__serbian"]["point"]))
    check("XL-large rms", "+%.3f and %.3f" % (a[K % "rms__punjabi"]["point"], a[K % "rms__serbian"]["point"]))
    check("XL-large script form", "%+.3f from large to XL" % a[K % "script_form"]["point"])
except Exception as e:
    n_fail += 1
    print("FAIL round 3/4 encoder checks: %s: %s" % (type(e).__name__, e))

try:
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from bootstrap import summarise as _s2
    names = {"xlm-roberta-base": "base", "xlm-roberta-large": "large", "muril-base-cased": "MuRIL"}
    BP = json.load(open(os.path.join(R3, "bracket_check_pumvr.json")))
    for m in names:
        c = _bj(os.path.join(R3, "bootstrap_pumvr_round3_%s.json" % m), m)["contrasts"]
        check("PuMVR %s trough" % names[m], f3(c["punjabi"]["trough"]["point"]))
        check("PuMVR %s de-voweled share" % names[m], f3(BP[m]["shares"]["script_form"]["point"]))
        check("PuMVR %s round-trip share" % names[m], f3(BP[m]["shares"]["script_form_rt"]["point"]))
        d = _np2.load(os.path.join(R3, "bootstrap_draws_pumvr_%s.npz" % m))
        tl, gl = int(_np2.argmin(d["point__punjabi"])), int(_np2.argmin(d["point__punjabi_gen"]))
        g = _s2(d["punjabi_gen"][gl] - d["punjabi"][tl],
                point=float(d["point__punjabi_gen"][gl] - d["point__punjabi"][tl]))
        check("PuMVR %s generated minus human" % names[m], "%.3f [%.3f, %.3f]" % (g["point"], g["ci95"][0], g["ci95"][1]))
    mu = BP["muril-base-cased"]["shares"]
    check("PuMVR MuRIL level 2 vs 1", "(%.3f against %.3f)" % (mu["script_form_dv2"]["point"], mu["script_form_dv1"]["point"]))
    nrev = sum(not v["monotone_point"] for r in BP.values() for v in r["per_seed"].values())
    check("PuMVR seeds with a reversed step", {4: "four of the nine"}.get(nrev, str(nrev)))
    ends = all(v["shares"]["script_form"]["point"] > v["shares"]["script_form_rt"]["point"]
               for r in BP.values() for v in r["per_seed"].values())
    if not ends:
        n_fail += 1
    print("%-4s %-46s %s" % ("ok" if ends else "FAIL", "PuMVR ends ordered in every seed", ends))

    Q = json.load(open(os.path.join(R3, "intervention_mcq_qwen2.5-7b.json")))["Qwen2.5-7B"]
    acc = Q["acc"]
    check("7B accuracy range", "%.3f to %.3f" % (min(acc.values()), max(acc.values())))
    check("7B Guru vs Shah p", "p = %.2f" % Q["mcnemar_guru_vs_shah"][2])
    check("7B gain", "(%.3f," % Q["gain"][0])
    check("7B gain p", "p = %.2f)" % Q["mcnemar_shah_vs_restored"][2])

    sib = json.load(open(os.path.join(RESULTS, "intervention_sib.json")))
    for m, keys in (("xlm-roberta-base", ("mcnemar_guru_vs_shah", "mcnemar_shah_vs_restored", "mcnemar_restored_vs_devowel")),
                    ("muril-base-cased", ("mcnemar_guru_vs_shah", "mcnemar_shah_vs_restored", "mcnemar_restored_vs_devowel"))):
        S = sib[m]["summary"]
        for k in keys:
            b, c = S[k][0], S[k][1]
            hi_, lo_ = max(b, c), min(b, c)
            if k == "mcnemar_restored_vs_devowel":
                hi_, lo_ = lo_, hi_
            if m == "xlm-roberta-base" and k == "mcnemar_guru_vs_shah":
                want = "%d are right in Gurmukhi only and %d" % (hi_, lo_)
            elif m == "xlm-roberta-base" and k == "mcnemar_shah_vs_restored":
                want = "%d items gained against %d lost" % (hi_, lo_)
            else:
                want = "%d against %d" % (hi_, lo_)
            check("%s %s counts" % (m, k), want)

    for tag, mid_s, mid_p in (("Qwen2.5-3B_poolmean", "0.619", "0.390"),):
        d = _np2.load(os.path.join(R3, "bootstrap_draws_flores_%s.npz" % tag))
        P, S = d["point__punjabi"], d["point__serbian"]
        L = len(P) // 2
        check("3B middle", "%.3f against %.3f in 3B" % (S[L], P[L]))
        _a = bool((S > P).all())
        if not _a:
            n_fail += 1
        print("%-4s %-46s %s" % ("ok" if _a else "FAIL", "3B Serbian above Punjabi at every layer", _a))
    d = _np2.load(os.path.join(R3, "bootstrap_draws_flores_gemma-2-2b_poolmean.npz"))
    x = d["point__serbian"] - d["point__punjabi"]
    pos = x > 0
    first = int(_np2.argmin(pos)) if not pos.all() else len(x)
    last = int(len(x) - _np2.argmin(pos[::-1]))
    check("Gemma split", "first %s and last %s of its %d depths" % (
        {10: "ten"}.get(first, first), {7: "seven"}.get(len(x) - last + 0, len(x) - last), len(x)))
    check("Gemma middle max", "by at most %.2f" % abs(x[first:last].min()))
except Exception as e:
    n_fail += 1
    print("FAIL PuMVR/decoder/intervention checks: %s: %s" % (type(e).__name__, e))

try:
    RS = json.load(open(os.path.join(R4, "resource_summary.json")))
    pj, sb, v = RS["punjabi"], RS["serbian"], RS["classifier_validation"]
    check("Gurmukhi letters", "%d million" % round(pj["gurmukhi_letters"] / 1e6))
    check("Shahmukhi range", "%.1f to %.1f million" % (pj["shahmukhi_letters_corrected"] / 1e6, pj["shahmukhi_letters_corrected_human_recall"] / 1e6))
    check("human-recall share", "%.1f%%" % (100 * pj["shahmukhi_over_gurmukhi_corrected_human_recall"]))
    hv = v["pumvr_human_shahmukhi"]
    check("human Shahmukhi recall", "%.1f%% as Punjabi, %.1f%% as Urdu" % (100 * hv["correct"], 100 * hv["wrong"]))
    check("Urdu letters", "%.1f billion" % (pj["urdu_letters"] / 1e9))
    check("Serbian Cyrillic", "%.1f billion" % (sb["cyrillic_letters_est"] / 1e9))
    check("Latin under sr", "%d million" % round(sb["latin_letters_under_sr_est"] / 1e6))
    check("Latin Serbo-Croatian", "%.1f billion" % (sb["latin_serbo_croatian_letters_est"] / 1e9))
    check("ekavian estimate", "%d million" % round(sb["ekavian_latin_letters_est"] / 1e6))
    check("Shahmukhi share", "%.1f%% to %.1f%%" % (100 * pj["shahmukhi_over_gurmukhi_corrected"], 100 * pj["shahmukhi_over_gurmukhi_raw"]))
    check("ekavian share", "%.1f%%" % (100 * sb["ekavian_latin_over_serbian_cyrillic"]))
    check("four times", {4: "four times"}[round(sb["latin_serbo_croatian_over_serbian_cyrillic"])])
    check("classifier recall", "%.1f%%" % (100 * v["pan_Shah"]["correct"]))
    check("classifier Urdu error", "%.2f%%" % (100 * v["urd_Arab"]["wrong"]))
    check("Croatian false ekavian", "%.1f%% of the decided Croatian" % (100 * sb["croatian_false_ekavian_rate"]))
    fr = sb["fractions_read"]
    check("fractions read", "%d%% and %d%%" % (round(100 * fr["sr"]), round(100 * fr["hr"])))
    RC = json.load(open(os.path.join(R3, "resource_counts.json")))
    w = RC["wikipedia"]
    check("pnb articles", "{:,}".format(w["pnb"]["articles"]))
    check("pa articles", "{:,}".format(w["pa"]["articles"]))
    vx = RC["vocabularies"]["xlm-roberta-base"]["entries_with_script"]
    check("XLM-R Gurmukhi vocab", "{:,}".format(vx["gurmukhi"]))
    check("XLM-R Arabic vocab", "{:,}".format(vx["arabic"]))
    check("Qwen Gurmukhi vocab", "(%d" % RC["vocabularies"]["Qwen/Qwen2.5-1.5B"]["entries_with_script"]["gurmukhi"])
    check("Gemma Gurmukhi vocab", "%d in Gemma-2" % RC["vocabularies"]["google/gemma-2-2b"]["entries_with_script"]["gurmukhi"])
except Exception as e:
    n_fail += 1
    print("FAIL resource checks: %s: %s" % (type(e).__name__, e))

try:
    ST = json.load(open(os.path.join(RESULTS, "stress_test.json")))
    sw = {x["param"]: x for x in ST["sweeps"]}
    check("stress gain", "by %.1f times" % sw["gain"]["consistency_vs_halving"])
    check("stress compress", "by %.2f times" % sw["compress"]["consistency_vs_halving"])
    check("stress tilt", "by %.2f times" % sw["tilt"]["consistency_vs_halving"])
    check("stress depth", "consistency by %.2f times" % sw["layer"]["consistency_vs_halving"])
    check("stress depth distances", "%.1f and %d times" % (sw["layer"]["shared_vs_halving"], round(sw["layer"]["rms_vs_halving"])))
except Exception as e:
    n_fail += 1
    print("FAIL stress checks: %s: %s" % (type(e).__name__, e))

print("\n%d failure(s)" % n_fail)
sys.exit(1 if n_fail else 0)
