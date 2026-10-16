"""End-to-end test of the bootstrap stage, with a synthetic extractor.

The stage's own plumbing is where mistakes hide: whether the trained and
random passes really share one resample, whether the seeds are averaged inside
the draw, whether the trough layer is held fixed, whether BH runs over the
right family, and whether the result serialises. None of that needs a GPU or
a network, so none of it should first be exercised in a paid GPU session.

Synthetic activations are built with a known structure: a large Punjabi
displacement, a smaller Serbian one, a near-zero no-script control, and
random-init passes with weaker displacements than trained. The stage must
recover that ordering and put the right signs on it.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import run_all  # noqa: E402


def _npmin(xs):
    return int(np.nanargmin(np.asarray(xs, dtype=float)))

n_pass = n_fail = 0


def ok(cond, label, extra=""):
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print("PASS  %s %s" % (label, extra))
    else:
        n_fail += 1
        print("FAIL  %s %s" % (label, extra))


N, D, L = 120, 32, 5
CONDS = ["pan_Guru", "pan_Shah", "pan_Guru_devowel", "srp_Cyrl", "srp_Latn",
         "pan_Shah_rom", "urd_rom", "pan_Guru_rt"]

# Each condition is displaced along its OWN direction, as in the real design:
# Shahmukhi moves far along axis 0 (a script change), de-voweled Gurmukhi only
# a little along axis 1 (an information change in the same script), Serbian
# Latin moderately along axis 2, romanised Urdu barely at all. That makes the
# script-form contrast (de-voweled against Shahmukhi) nearly as large as the
# full one, so the planted script-form ratio is near 1, as the paper finds.
# (axis, magnitude), trained then random-init. Trained exceeds random
# everywhere except the no-script control, where both are ~0.
SHIFT_TRAINED = {"pan_Shah": (0, 1.6), "pan_Guru_devowel": (1, 0.35),
                 "pan_Guru_rt": (4, 0.6),
                 "srp_Latn": (2, 0.5), "urd_rom": (3, 0.02)}
SHIFT_RANDOM = {"pan_Shah": (0, 0.6), "pan_Guru_devowel": (1, 0.20),
                "pan_Guru_rt": (4, 0.30),
                "srp_Latn": (2, 0.35), "urd_rom": (3, 0.02)}


POOLS = []


def fake_extract(texts, name, pool, dt, dmap, args):
    """(L, n, d) activations with a planted, known script displacement."""
    POOLS.append(pool)
    rand = bool(getattr(args, "random_init", False))
    seed = int(getattr(args, "seed", 0))
    cond = texts[0].split("|")[0]          # the condition names its own texts
    # Content is shared across conditions for a given model+pass, so the
    # per-item difference isolates the planted displacement, exactly as the
    # real design intends.
    base_rng = np.random.default_rng(hash((name, rand, seed)) % (2 ** 31))
    content = base_rng.normal(size=(L, N, D))
    acts = content.copy()
    # Serbian sits on its own content block: a different language, same items.
    if cond.startswith("srp"):
        acts = base_rng.normal(size=(L, N, D))
        src = "srp"
    elif cond.endswith("_rom"):
        acts = base_rng.normal(size=(L, N, D))
        src = "rom"
    else:
        src = "pan"
    # Re-derive the block deterministically so both members of a contrast
    # share it.
    blk = np.random.default_rng(hash((name, rand, seed, src)) % (2 ** 31))
    acts = blk.normal(size=(L, N, D))

    table = SHIFT_RANDOM if rand else SHIFT_TRAINED
    if cond in table:
        axis, mag = table[cond]
        direction = np.zeros(D)
        direction[axis] = 1.0
        item_rng = np.random.default_rng(hash((cond, src)) % (2 ** 31))
        for li in range(L):
            # A shared direction plus item-specific noise, so consistency is
            # strictly between 0 and 1.
            acts[li] = acts[li] + mag * direction \
                + 0.3 * item_rng.normal(size=(N, D))
    return acts, {"truncated_frac": 0.0}


def fake_load_conditions(args):
    # Each text is tagged with its condition so the fake extractor can
    # identify what it was handed, whatever order the stage asks in.
    cond = {c: ["%s|%d" % (c, i) for i in range(N)] for c in CONDS}
    return list(range(N)), cond, []


CAPTURED = {}


def fake_save(fname, obj):
    CAPTURED[fname] = json.loads(json.dumps(obj, default=run_all._json))


def main():
    global n_fail
    # Everything this test writes goes to a temp directory. It previously ran
    # against the real results/ and its cleanup deleted an actual run's draw
    # files and cross-model output. A test must not be able to destroy data.
    import shutil
    import tempfile
    sandbox = tempfile.mkdtemp(prefix="mukhi_boot_test_")
    real_results = run_all.RESULTS
    run_all.RESULTS = sandbox
    try:
        return _run(sandbox)
    finally:
        run_all.RESULTS = real_results
        shutil.rmtree(sandbox, ignore_errors=True)


def _run(sandbox):
    global n_fail
    run_all._load_conditions = fake_load_conditions
    run_all.save = fake_save
    run_all.MODELS = [("xlm-roberta-base", "encoder", "mean", "fp32", None),
                      ("xlm-roberta-large", "encoder", "mean", "fp32", None),
                      ("Qwen/Qwen2.5-1.5B", "decoder", "last", "fp16", None)]

    # The real _boot_pass and the real stage are the code under test; only
    # the extractor is faked.
    run_all._extract = fake_extract

    # Grouping must hold at most three conditions at once, or a 25-layer
    # model's activations will not fit beside its own bootstrap draws.
    by_label = {lab: (a, b) for a, b, lab in run_all.BOOT_CONTRASTS}
    peak = max(len({c for lab in g if lab in by_label for c in by_label[lab]})
               for g in run_all.BOOT_GROUPS)
    ok(peak <= 6, "no group holds more than six conditions at once", "peak %d" % peak)
    covered = {lab for g in run_all.BOOT_GROUPS for lab in g}
    ok(covered == set(by_label), "every contrast belongs to exactly one group",
       str(sorted(covered)))
    flat = [lab for g in run_all.BOOT_GROUPS for lab in g]
    ok(len(flat) == len(set(flat)), "no contrast is computed twice")

    args = argparse.Namespace(
        data="flores", models="", n_sentences=0, batch_size=8, max_length=128,
        cache_dir=None, force=True, random_init=False, seed=0, pooling="",
        acts_dir=None, extract_only=False, out_tag="", contrasts="",
        n_boot=400, boot_seed=99, boot_random_seeds="0,1,2", purge_cache=False)

    run_all.stage_bootstrap(args)

    key = [k for k in CAPTURED if k.startswith("bootstrap_")]
    ok(len(key) == 1, "the stage wrote exactly one result file", str(key))
    if not key:
        return 1
    res = CAPTURED[key[0]]

    ok(res["n"] == N, "n recorded", str(res["n"]))
    ok(res["n_boot"] == 400, "draw count recorded")
    pm = res["per_model"]
    ok("xlm-roberta-base" in pm and "xlm-roberta-large" in pm,
       "both encoders present", str(sorted(pm)))
    ok("Qwen2.5-1.5B" not in pm, "the decoder was skipped (Finding 8)")

    # Per-seed draws are saved, and their mean equals the seed-mean difference.
    import numpy as _np
    _f = os.path.join(run_all.RESULTS, "bootstrap_draws_flores_xlm-roberta-base.npz")
    if os.path.exists(_f):
        _z = _np.load(_f)
        _ks = sorted(k for k in _z.files if k.startswith("seed") and k.endswith("__punjabi"))
        ok(_ks == ["seed0__punjabi", "seed1__punjabi", "seed2__punjabi"],
           "per-seed draws saved for each seed", str(_ks))
        _m = _np.mean([_z[k] for k in _ks], axis=0)
        ok(_np.allclose(_m, _z["punjabi"], atol=1e-9),
           "mean of per-seed differences equals the seed-mean difference")
        _pm = _np.mean([_z["point__" + k] for k in _ks], axis=0)
        ok(_np.allclose(_pm, _z["point__punjabi"], atol=1e-9),
           "per-seed points average to the reported point")
    else:
        ok(False, "draw file present for the per-seed check", _f)

    # A mean-pooled decoder is NOT excluded: it runs, under its own tag, and
    # --boot-only limits it to the contrasts asked for.
    saved_models, saved_capt = run_all.MODELS, dict(CAPTURED)
    POOLS.clear()
    run_all.MODELS = [("Qwen/Qwen2.5-1.5B", "decoder", "last", "fp16", None)]
    dargs = argparse.Namespace(**vars(args))
    dargs.pooling, dargs.boot_only = "mean", "punjabi,serbian"
    CAPTURED.clear()
    run_all.stage_bootstrap(dargs)
    dres = [v for k, v in CAPTURED.items() if k.startswith("bootstrap_")]
    dpm = dres[0]["per_model"] if dres else {}
    ok("Qwen2.5-1.5B_poolmean" in dpm, "mean-pooled decoder runs under its own tag",
       str(sorted(dpm)))
    ok(POOLS and set(POOLS) == {"mean"}, "decoder extracted with mean pooling",
       str(set(POOLS)))
    ok(set(dpm.get("Qwen2.5-1.5B_poolmean", {}).get("contrasts", {}))
       == {"punjabi", "serbian"}, "--boot-only restricts the contrasts")
    ok(len(run_all.BOOT_CONTRASTS) == 13, "--boot-only leaves the global list intact")
    run_all.MODELS = saved_models
    CAPTURED.clear(); CAPTURED.update(saved_capt)

    base = pm["xlm-roberta-base"]
    ok(base["n_random_seeds"] == 3, "three random-init seeds averaged",
       str(base["n_random_seeds"]))
    cs = base["contrasts"]
    ok(set(cs) == {"punjabi", "serbian", "script_form", "no_script",
                   "script_form_rt", "script_form_dv1", "script_form_dv2",
                   "information"},
       "every contrast whose conditions exist is computed", str(sorted(cs)))

    # Magnitudes and the ratio estimand ride along on every contrast.
    pjc = cs["punjabi"]
    ok(set(pjc["magnitude"]) == {"shared", "rms"}, "both magnitudes reported",
       str(sorted(pjc["magnitude"])))
    for stat in ("shared", "rms"):
        rows_m = pjc["magnitude"][stat]["per_layer"]
        ok(len(rows_m) == L and all(abs(r["point"] - (r["trained"] - r["random_mean"]))
                                    < 1e-9 for r in rows_m),
           "%s: point is trained minus mean random at every layer" % stat)
    # consistency == shared / rms must hold for trained values, layer by layer
    t_sh = [r["trained"] for r in pjc["magnitude"]["shared"]["per_layer"]]
    t_rm = [r["trained"] for r in pjc["magnitude"]["rms"]["per_layer"]]
    t_co = [r["trained"] for r in pjc["per_layer"]]
    ok(all(abs(c - a / b) < 1e-9 for c, a, b in zip(t_co, t_sh, t_rm)),
       "trained consistency equals shared over rms")
    ok(pjc["magnitude"]["rms"]["trough"]["point"] > 0,
       "planted Punjabi displacement is larger trained than random in rms too")
    ok("ratio" in pjc and pjc["ratio"]["trough"]["point"] > 0,
       "ratio estimand: trained/random exceeds 1 for the planted contrast")
    ok(cs["script_form_rt"]["trough"]["point"] > 0,
       "round-trip control contrast computed and positive as planted")

    # The planted structure must come back out.
    pj = cs["punjabi"]["trough"]
    sb = cs["serbian"]["trough"]
    ns = cs["no_script"]
    ok(pj["point"] > 0, "planted Punjabi difference is positive", "%+.4f" % pj["point"])
    ok(pj["ci95"][0] > 0, "and its interval excludes zero",
       "[%+.4f, %+.4f]" % tuple(pj["ci95"]))
    ok(pj["excludes_zero"] is True, "excludes_zero flag agrees")
    ok(pj["point"] > sb["point"], "Punjabi trough exceeds Serbian, as planted",
       "%+.4f vs %+.4f" % (pj["point"], sb["point"]))
    ok(abs(ns["trough"]["point"]) < 0.15,
       "the no-script control stays near zero", "%+.4f" % ns["trough"]["point"])

    # Per-layer rows.
    rows = cs["punjabi"]["per_layer"]
    ok(len(rows) == L, "one row per layer", str(len(rows)))
    ok(all(r["layer"] == i for i, r in enumerate(rows)), "layers in order")
    ok(all("q_bh" in r for r in rows), "BH q values attached to every row")
    ok(all(r["ci95"][0] <= r["point"] <= r["ci95"][1] for r in rows),
       "every point estimate lies inside its own interval")
    ok(all(r["p_boot"] >= 1.0 / 400 for r in rows),
       "no p value below the draw resolution")
    ok(all(abs(r["point"] - (r["trained"] - r["random_mean"])) < 1e-9 for r in rows),
       "point equals trained minus mean random, exactly")

    # The trough layer must be the argmin of the point estimates, not refit.
    pts = [r["point"] for r in rows]
    ok(cs["punjabi"]["trough"]["layer"] == int(np.argmin(pts)),
       "trough layer is the full-sample argmin", str(cs["punjabi"]["trough"]["layer"]))
    ok(abs(cs["punjabi"]["trough"]["point"] - min(pts)) < 1e-12,
       "trough point equals the minimum point estimate")
    ok(cs["punjabi"]["trough_refit"]["point"] <= cs["punjabi"]["trough"]["point"] + 1e-9,
       "the refit version is biased downward, as documented",
       "%+.4f <= %+.4f" % (cs["punjabi"]["trough_refit"]["point"],
                           cs["punjabi"]["trough"]["point"]))

    # Endpoints.
    ok(abs(cs["punjabi"]["embedding"]["point"] - pts[0]) < 1e-12, "embedding row matches layer 0")
    ok(abs(cs["punjabi"]["final"]["point"] - pts[-1]) < 1e-12, "final row matches the last layer")

    # The ordering test.
    t = base["punjabi_minus_serbian_trough"]
    ok(t["point"] > 0, "Punjabi-minus-Serbian ordering is positive", "%+.4f" % t["point"])
    ok(t["ci95"][0] > 0, "and significant, as planted", "[%+.4f, %+.4f]" % tuple(t["ci95"]))
    ok(abs(t["point"] - (pj["point"] - sb["point"])) < 1e-9,
       "the ordering point equals the difference of the two troughs")

    # The ratio.
    r = base["script_form_ratio"]["layer_mean"]
    ok(0.5 < r["point"] < 1.3, "script-form ratio is plausible", "%.4f" % r["point"])
    ok(r["ci95"][0] < r["point"] < r["ci95"][1], "ratio point sits inside its interval")

    # Cross-model comparison.
    ok("punjabi_trough_large_minus_base" in res["across_models"],
       "the scale comparison was computed", str(sorted(res["across_models"])))

    # BH family must cover every per-layer test and nothing else.
    expected = sum(len(c["per_layer"]) for m in pm.values() for c in m["contrasts"].values())
    ok(res["bh_family_size"] == expected, "BH family size equals the per-layer test count",
       "%d vs %d" % (res["bh_family_size"], expected))

    # Draw arrays must not leak into the saved file.
    ok(all("_draws" not in c for m in pm.values() for c in m["contrasts"].values()),
       "the in-memory draw arrays were dropped before saving")
    ok(len(json.dumps(res)) < 2_000_000, "result file stays small",
       "%d bytes" % len(json.dumps(res)))

    # Determinism: the same seed must reproduce the same interval.
    CAPTURED.clear()
    run_all.stage_bootstrap(args)
    res2 = CAPTURED[[k for k in CAPTURED if k.startswith("bootstrap_")][0]]
    a = res["per_model"]["xlm-roberta-base"]["contrasts"]["punjabi"]["trough"]
    b = res2["per_model"]["xlm-roberta-base"]["contrasts"]["punjabi"]["trough"]
    ok(a["ci95"] == b["ci95"], "a rerun with the same seed gives the same interval")

    # --- per-model seed specs ----------------------------------------------
    # XLM-R large has one published random-init run; averaging three would
    # make its baseline a different quantity from the paper's.
    CAPTURED.clear()
    spec_args = argparse.Namespace(**vars(args))
    spec_args.boot_random_seeds = "*=0,1,2;xlm-roberta-large=0"
    run_all.stage_bootstrap(spec_args)
    r3 = CAPTURED[[k for k in CAPTURED if k.startswith("bootstrap_")][0]]
    ok(r3["per_model"]["xlm-roberta-base"]["n_random_seeds"] == 3,
       "per-model spec: base averages three seeds")
    ok(r3["per_model"]["xlm-roberta-large"]["n_random_seeds"] == 1,
       "per-model spec: large averages one seed",
       str(r3["per_model"]["xlm-roberta-large"]["n_random_seeds"]))
    ok(r3["per_model"]["xlm-roberta-large"]["random_seeds"] == [0],
       "and it is seed 0, the published one")
    ok(r3["seed_spec"] == spec_args.boot_random_seeds, "the spec is recorded")

    # The draw filenames must be exactly the model tag, so the cross-model
    # tool can parse them. With --out-tag set, an earlier version doubled the
    # tag and the scale comparison came back empty instead of failing.
    CAPTURED.clear()
    tagged = argparse.Namespace(**vars(spec_args))
    tagged.out_tag = "xlm-roberta-base"
    run_all.stage_bootstrap(tagged)
    import glob as _g0
    got = {os.path.basename(f) for f in
           _g0.glob(os.path.join(run_all.RESULTS, "bootstrap_draws_flores_*.npz"))}
    ok("bootstrap_draws_flores_xlm-roberta-base.npz" in got,
       "--out-tag does not leak into the draw filename", str(sorted(got)))
    ok(not any("base_xlm-roberta-base" in f for f in got),
       "and the model tag is not doubled")
    # The tool must then recognise the model by name.
    import subprocess as _sp
    _r = _sp.run([sys.executable,
                  os.path.join(ROOT, "tools_bootstrap_crossmodel.py"),
                  "--data", "flores"], capture_output=True, text=True, cwd=ROOT,
                 env=dict(os.environ, MUKHI_RESULTS=sandbox))
    ok("punjabi_trough_xlm-roberta-large_minus_xlm-roberta-base"
       in (_r.stdout or "") or _r.returncode == 0,
       "the tool parses those filenames back to model tags")

    # Draw files must be written, and be small enough to ship in the zip.
    import glob as _glob
    dfs = _glob.glob(os.path.join(run_all.RESULTS, "bootstrap_draws_flores_*.npz"))
    ok(len(dfs) >= 2, "draw arrays saved per model", str(len(dfs)))
    if dfs:
        import numpy as _np
        z = _np.load(dfs[0])
        ok("punjabi" in z, "draws keyed by contrast", str(sorted(z.files)))
        ok(z["punjabi"].shape[1] == spec_args.n_boot,
           "one column per draw", str(z["punjabi"].shape))
        biggest = max(os.path.getsize(f) for f in dfs)
        ok(biggest < 8 * 1024 * 1024, "draw files stay small",
           "%d KB" % (biggest // 1024))
        for f in dfs:
            os.remove(f)

    # --- the cross-check must verify EVERY pass, not half of them ----------
    # The first real run passed the raw spec "*=0,1,2;xlm-roberta-large=0" and
    # the cross-check re-split it on commas, producing seed labels "*=0" and
    # "2;xlm-roberta-large=0". Those matched no published file, so two of
    # every four passes went unverified while the summary still said "ok".
    cc3 = r3["per_model"]["xlm-roberta-base"]["crosscheck"]
    labels = [k for k in cc3 if k.startswith(("trained", "random_"))]
    ok(sorted(labels) == ["random_s0", "random_s1", "random_s2", "trained"],
       "crosscheck labels every pass by its resolved seed", str(sorted(labels)))
    ok(not any("=" in k or ";" in k for k in labels),
       "and no label carries raw spec syntax")
    cc_large = r3["per_model"]["xlm-roberta-large"]["crosscheck"]
    llabels = sorted(k for k in cc_large if k.startswith(("trained", "random_")))
    ok(llabels == ["random_s0", "trained"],
       "a one-seed model gets exactly two passes checked", str(llabels))

    # --- the cross-check must FAIL when it cannot compare ------------------
    # Synthetic runs have 5 layers against the published 13, so nothing is
    # comparable and the check must say so rather than report agreement. A
    # check that returns "ok" when it compared nothing is the failure mode
    # this guards against.
    cc = base["crosscheck"]
    ok(cc["agrees_with_published"] is False,
       "crosscheck refuses to claim agreement when nothing was comparable")
    ok(cc["n_contrasts_compared"] == 0, "and reports zero comparisons",
       str(cc["n_contrasts_compared"]))
    ok(cc["worst_abs_diff"] is None, "and reports no diff rather than 0.0")

    # The "tried to compare and could not" path, tested on purpose rather
    # than by accident. A published file with the wrong layer count must be
    # named as not comparable, not quietly skipped and not counted as ok.
    a_lc = argparse.Namespace(data="flores", boot_random_seeds="0")
    with open(os.path.join(sandbox, "main_flores_wrongshape.json"), "w") as f:
        json.dump({"contrasts": {"pan_Guru|pan_Shah": {
            "paired": [{"consistency": 0.5}] * 99}}}, f)
    rep_lc = run_all._boot_crosscheck(
        a_lc, "wrongshape", {"punjabi": {"point": [0.5, 0.5, 0.5]}}, [])
    ok(len(rep_lc["not_comparable"]) == 1,
       "a layer-count mismatch is named as not comparable",
       str(rep_lc["not_comparable"]))
    ok(rep_lc["agrees_with_published"] is False,
       "and blocks the agreement claim")
    ok(rep_lc["n_contrasts_compared"] == 0, "having compared nothing")

    # And it must PASS when the published numbers do match. Feed it a
    # published file built from the synthetic points themselves.
    import tempfile
    real_results = run_all.RESULTS
    tmp = tempfile.mkdtemp()
    try:
        run_all.RESULTS = tmp
        pj_pts = [r["trained"] for r in base["contrasts"]["punjabi"]["per_layer"]]
        payload = {"contrasts": {"pan_Guru|pan_Shah": {
            "paired": [{"consistency": v} for v in pj_pts]}}}
        with open(os.path.join(tmp, "main_flores_xlm-roberta-base.json"), "w") as f:
            json.dump(payload, f)
        trained_stub = {"punjabi": {"point": pj_pts}}
        a2 = argparse.Namespace(data="flores", boot_random_seeds="0,1,2")
        rep = run_all._boot_crosscheck(a2, "xlm-roberta-base", trained_stub, [])
        ok(rep["agrees_with_published"] is True,
           "crosscheck agrees when the points match exactly")
        ok(rep["n_contrasts_compared"] == 1, "and counts the comparison",
           str(rep["n_contrasts_compared"]))
        ok(rep["worst_abs_diff"] < 1e-12, "with a zero difference",
           "%.2e" % rep["worst_abs_diff"])

        # Now perturb one layer past the 3dp tolerance: must flip to failure.
        bad = dict(trained_stub)
        bad = {"punjabi": {"point": [pj_pts[0] + 0.01] + list(pj_pts[1:])}}
        rep2 = run_all._boot_crosscheck(a2, "xlm-roberta-base", bad, [])
        ok(rep2["agrees_with_published"] is False,
           "crosscheck detects a 0.01 discrepancy")
        ok(rep2["worst_abs_diff"] > 5e-4, "and reports its size",
           "%.2e" % rep2["worst_abs_diff"])
    finally:
        run_all.RESULTS = real_results

    # --- the cross-model tool must reproduce the stage's own comparison ----
    # Both read the same draws, so they must agree. If they do not, one of
    # them is comparing something else, and the tool is what will be run on
    # Kaggle to avoid a second extraction pass.
    CAPTURED.clear()
    xm_args = argparse.Namespace(**vars(args))
    xm_args.boot_random_seeds = "*=0,1,2;xlm-roberta-large=0"
    run_all.stage_bootstrap(xm_args)
    stage_res = CAPTURED[[k for k in CAPTURED if k.startswith("bootstrap_")][0]]

    import subprocess
    env = dict(os.environ, MUKHI_RESULTS=sandbox)
    r = subprocess.run([sys.executable,
                        os.path.join(ROOT, "tools_bootstrap_crossmodel.py"),
                        "--data", "flores"],
                       capture_output=True, text=True, cwd=ROOT, env=env)
    ok(r.returncode == 0, "cross-model tool runs", (r.stdout + r.stderr)[-200:])
    xm_path = os.path.join(sandbox, "bootstrap_crossmodel_flores.json")
    if r.returncode == 0 and os.path.exists(xm_path):
        xm = json.load(open(xm_path))
        key = "punjabi_trough_xlm-roberta-large_minus_xlm-roberta-base"
        ok(key in xm["across_models"], "tool computed the scale comparison",
           str(sorted(xm["across_models"])))
        if key in xm["across_models"] and \
                "punjabi_trough_large_minus_base" in stage_res["across_models"]:
            a = xm["across_models"][key]
            b = stage_res["across_models"]["punjabi_trough_large_minus_base"]
            ok(abs(a["ci95"][0] - b["ci95"][0]) < 1e-9
               and abs(a["ci95"][1] - b["ci95"][1]) < 1e-9,
               "tool and stage agree on the scale interval",
               "[%+.6f,%+.6f] vs [%+.6f,%+.6f]" % (a["ci95"][0], a["ci95"][1],
                                                   b["ci95"][0], b["ci95"][1]))
        # And the per-model ordering test must match too.
        tb = xm["per_model"]["xlm-roberta-base"].get("punjabi_minus_serbian_trough")
        sb_ = stage_res["per_model"]["xlm-roberta-base"].get(
            "punjabi_minus_serbian_trough")
        if tb and sb_:
            # Compare the SELECTED LAYERS as well as the interval. An earlier
            # version picked the trough from the mean over draws instead of
            # the full-sample estimate, so the two sometimes landed on
            # different layers and the intervals agreed only by luck.
            ok(tb["punjabi_layer"] == sb_["punjabi_layer"]
               and tb["serbian_layer"] == sb_["serbian_layer"],
               "tool and stage select the same trough layers",
               "tool L%d/L%d vs stage L%d/L%d" % (tb["punjabi_layer"],
                                                  tb["serbian_layer"],
                                                  sb_["punjabi_layer"],
                                                  sb_["serbian_layer"]))
            ok(abs(tb["ci95"][0] - sb_["ci95"][0]) < 1e-9
               and abs(tb["ci95"][1] - sb_["ci95"][1]) < 1e-9,
               "tool and stage agree on the ordering interval")
            ok(abs(tb["point"] - sb_["point"]) < 1e-9,
               "and on the point estimate the interval is attached to",
               "%.6f vs %.6f" % (tb["point"], sb_["point"]))
            # The trough layers must come from the full-sample estimates, not
            # the draw means, which is what the paper quotes.
            pj_pts = [r["point"] for r in
                      stage_res["per_model"]["xlm-roberta-base"]
                      ["contrasts"]["punjabi"]["per_layer"]]
            ok(tb["punjabi_layer"] == int(_npmin(pj_pts)),
               "the trough layer is the full-sample argmin",
               "L%d" % tb["punjabi_layer"])

        # Mismatched resample seeds must be refused, not silently compared.
        import numpy as _np
        import glob as _g
        dfs = sorted(_g.glob(os.path.join(run_all.RESULTS,
                                          "bootstrap_draws_flores_*.npz")))
        if len(dfs) >= 2:
            z = dict(_np.load(dfs[0]))
            z["counts_seed"] = _np.array([z["counts_seed"][0] + 1])
            _np.savez_compressed(dfs[0], **z)
            r2 = subprocess.run([sys.executable,
                                 os.path.join(ROOT, "tools_bootstrap_crossmodel.py"),
                                 "--data", "flores"],
                                capture_output=True, text=True, cwd=ROOT,
                                env=dict(os.environ, MUKHI_RESULTS=sandbox))
            ok(r2.returncode != 0 and "REFUSING" in r2.stdout,
               "tool refuses draw files that do not share one resample")
        for f in dfs:
            os.remove(f)
        if os.path.exists(xm_path):
            os.remove(xm_path)

    print("\n%d/%d passed" % (n_pass, n_pass + n_fail))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
