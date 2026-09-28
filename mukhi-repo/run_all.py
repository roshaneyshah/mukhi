#!/usr/bin/env python3
"""
MUKHI end-to-end runner.

Stages (run in this order on Kaggle / Colab):

  smoke               synthetic planted-structure check; no network
  translit_eval       both rule transliterators scored on PuMVR; needs GitHub
  g2s_compare         try the SLPG fairseq model; keep it only if it beats rules
  conditions          FLORES+ and SIB-200 conditions (HF token for FLORES+)
  fertility           tokeniser premium per condition per model; no GPU
  audit / score_audit human check of the generated Shahmukhi
  main                CKA + probes + paired displacement, every contrast
  intervention        SIB-200 frozen-feature transfer (proposal step 5)
  intervention_mcq    PuMVR option scoring on human Shahmukhi (decoders)
  report              aggregate every JSON into results/REPORT.md

`--data pumvr` runs `main` and `fertility` on the 1,000 human-written PuMVR
sentences instead of FLORES (robustness check; runs offline given GitHub).
`--models` accepts HF ids or local directories.
"""
import argparse
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(RESULTS, "figures")
PUMVR_DIR = os.path.join(ROOT, "data", "pumvr")

# (hf id, kind, pooling, dtype hint, device_map)
# MuRIL and XLM-R base share d=768: their comparison is not confounded by the
# dimension-dependent CKA bias (FINDING_01). Gemma-2 overflows in fp16 on T4.
MODELS = [
    ("xlm-roberta-base",           "encoder", "mean", "fp16", None),
    ("google/muril-base-cased",    "encoder", "mean", "fp16", None),
    ("xlm-roberta-large",          "encoder", "mean", "fp16", None),
    ("Qwen/Qwen2.5-1.5B",          "decoder", "last", "fp16", None),
    ("google/gemma-2-2b",          "decoder", "last", "fp16", None),
    ("Qwen/Qwen2.5-3B",            "decoder", "last", "fp16", None),
    ("CohereLabs/aya-expanse-8b", "decoder", "last", "fp16", "auto"),
]


def _json(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _json(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_json(v) for v in obj]
    return obj


def save(name, payload):
    os.makedirs(RESULTS, exist_ok=True)
    p = os.path.join(RESULTS, name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(_json(payload), f, indent=2, ensure_ascii=False)
    print(f"  -> {p}")
    return p


def load(name):
    return json.load(open(os.path.join(RESULTS, name), encoding="utf-8"))


def selected_models(args):
    if not args.models:
        return MODELS
    out = []
    for m in args.models.split(","):
        m = m.strip()
        hit = [x for x in MODELS if x[0] == m]
        if hit:
            out.extend(hit)
            continue
        kind = "encoder"                  # local dir or unlisted id: infer
        cfgp = os.path.join(m, "config.json")
        if os.path.exists(cfgp):
            arch = " ".join(json.load(open(cfgp)).get("architectures", []))
            if any(k in arch for k in ("GPT", "CausalLM", "Llama", "Qwen", "Gemma")):
                kind = "decoder"
        out.append((m, kind, "last" if kind == "decoder" else "mean", "fp32", None))
    return out


def purge_model_cache(name, cache_dir):
    """Delete one model's cached weights. Kaggle gives 20GB of working disk and
    the large models are 3 to 10GB each, so a multi-model run fills it."""
    import shutil
    root = cache_dir or os.path.expanduser("~/.cache/huggingface/hub")
    slug = "models--" + name.replace("/", "--")
    for base in (root, os.path.join(root, "hub")):
        p = os.path.join(base, slug)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
            print(f"    purged cached weights: {p}")
            return True
    return False


def _dtype(hint):
    import torch
    if not torch.cuda.is_available():
        return torch.float32
    return torch.float32 if hint == "fp32" else torch.float16


def pumvr_json():
    from pumvr import fetch_pumvr
    return fetch_pumvr(PUMVR_DIR)


# ---------------------------------------------------------------------------
def stage_smoke(args):
    from synthetic import build_synthetic
    from analysis import (layerwise_cka, layerwise_probe,
                          layerwise_paired_displacement, crossover_layer, summarise_run)
    from plots import plot_cka_curve, plot_selectivity, plot_paired_displacement

    n = min(args.n_sentences or 300, 400)
    print(f"[smoke] synthetic model, n={n}")
    ag, ash, au = build_synthetic(n_items=n, converges=True, seed=0)
    cka = layerwise_cka(ag, ash, n_permutations=50, seed=0)
    pid = [f"p{i}" for i in range(n)] * 2
    y = np.array([0] * n + [1] * n)
    sl = layerwise_probe([ag, ash], y, pid, seed=0)
    ll = layerwise_probe([ash, au], y, [f"q{i}" for i in range(n)] * 2, seed=0)
    pd_ = layerwise_paired_displacement(ag, ash, n_permutations=50, seed=0)
    xo = crossover_layer(script_sel=[r["selectivity"] for r in sl],
                         language_sel=[r["selectivity"] for r in ll])
    os.makedirs(FIGS, exist_ok=True)
    plot_cka_curve(cka, f"{FIGS}/smoke_cka.png", "synthetic", "Gurmukhi vs Shahmukhi")
    plot_selectivity(sl, ll, f"{FIGS}/smoke_selectivity.png", "synthetic", xo)
    plot_paired_displacement(pd_, f"{FIGS}/smoke_paired.png", "synthetic")
    summ = summarise_run(cka, sl, ll)
    print(f"  crossover layer: {summ['crossover_layer']}   (planted: mid-stack)")
    save("smoke.json", {"summary": summ})


# ---------------------------------------------------------------------------
def stage_translit_eval(args):
    """Both rule systems against human PuMVR; held-out split scored once."""
    from pumvr import load_pumvr_pairs
    from translit import rule_based_shah_to_gur
    from translit_ctx import ctx_shah_to_gur
    from translit_g2s import gur_to_shah, normalise_shah
    from translit_eval import (cer, split_pairs, bootstrap_cer_ci,
                               error_decomposition, levenshtein)

    S, W, _ = load_pumvr_pairs(pumvr_json())
    dev, test = split_pairs(W)
    out = {"n_sentences": len(S), "n_word_pairs_unique": len(dev) + len(test)}

    for name, fn in (("s2g_naive", rule_based_shah_to_gur), ("s2g_context", ctx_shah_to_gur)):
        h = [fn(s) for g, s in test]
        r = [g for g, s in test]
        lo, hi = bootstrap_cer_ci(h, r)
        out[f"{name}_test_words"] = {**cer(h, r), "ci95": [lo, hi]}
        out[f"{name}_sentences"] = cer([fn(x["shahmukhi"]) for x in S],
                                       [x["gurmukhi"] for x in S])

    dec = error_decomposition([ctx_shah_to_gur(x["shahmukhi"]) for x in S],
                              [x["gurmukhi"] for x in S])
    out["s2g_context_unwritten_share"] = dec["unwritten_share"]

    def cer_s(h, r):
        e = sum(levenshtein(normalise_shah(a), normalise_shah(b)) for a, b in zip(h, r))
        return e / max(sum(len(normalise_shah(b)) for b in r), 1)
    out["g2s_rules_test_words_cer"] = cer_s([gur_to_shah(g) for g, s in test],
                                            [s for g, s in test])
    out["g2s_rules_sentences_cer"] = cer_s([gur_to_shah(x["gurmukhi"]) for x in S],
                                           [x["shahmukhi"] for x in S])
    for k, v in out.items():
        print(f"  {k}: {v}")
    save("translit_eval.json", out)


def stage_g2s_compare(args):
    """Keep the SLPG fairseq model only if it beats the rules on PuMVR."""
    from translit import load_slpg_fairseq, score_on_pumvr
    from translit_g2s import gur_to_shah
    pj = pumvr_json()
    rules = score_on_pumvr(gur_to_shah, pj, "g2s")
    choice, nmt = "rules", None
    try:
        fn = load_slpg_fairseq("g2s", cache_dir=args.cache_dir)
        nmt = score_on_pumvr(fn, pj, "g2s")
        choice = "nmt" if nmt < rules else "rules"
    except Exception as e:
        print(f"  fairseq model unavailable: {type(e).__name__}: {str(e)[:160]}")
    print(f"  PuMVR CER  rules={rules:.4f}  nmt={nmt}  -> using {choice}")
    save("g2s_choice.json", {"rules_cer": rules, "nmt_cer": nmt, "choice": choice})


def _g2s_fn(args):
    p = os.path.join(RESULTS, "g2s_choice.json")
    if os.path.exists(p) and load("g2s_choice.json")["choice"] == "nmt":
        from translit import load_slpg_fairseq
        print("  Shahmukhi generator: SLPG fairseq (won g2s_compare)")
        return load_slpg_fairseq("g2s", cache_dir=args.cache_dir)
    from translit_g2s import gur_to_shah
    print("  Shahmukhi generator: rules (5.6% held-out CER)")
    return gur_to_shah


# ---------------------------------------------------------------------------
def stage_conditions(args):
    """Build and cache every condition text set once, so later stages never
    regenerate (and can never silently differ)."""
    from conditions import load_flores, build_conditions, devowel_gurmukhi
    from translit_ctx import ctx_shah_to_gur
    from datasets import load_dataset
    g2s = _g2s_fn(args)

    ids, fl = load_flores(cache_dir=args.cache_dir)
    cond = build_conditions(ids, fl, gur_to_shah=g2s)
    save("conditions_flores.json", {"ids": ids, "conditions": cond})

    sib = {}
    for split in ("train", "validation", "test"):
        g = load_dataset("Davlan/sib200", "pan_Guru", split=split, cache_dir=args.cache_dir)
        texts = [r["text"] for r in g]
        shah = [g2s(t) for t in texts]
        sib[split] = {"index_id": [r["index_id"] for r in g],
                      "label": [r["category"] for r in g],
                      "pan_Guru": texts,
                      "pan_Guru_devowel": [devowel_gurmukhi(t) for t in texts],
                      "pan_Shah": shah,
                      "pan_Shah_to_Guru": [ctx_shah_to_gur(t) for t in shah]}
    save("conditions_sib.json", sib)
    print(f"  FLORES items={len(ids)} conditions={sorted(cond)}")
    print(f"  SIB sizes={ {k: len(v['label']) for k, v in sib.items()} }")


def _pumvr_conditions():
    """Human-written parallel Punjabi (robustness set)."""
    from pumvr import load_pumvr_pairs
    from conditions import devowel_gurmukhi, romanize
    S, _, _ = load_pumvr_pairs(pumvr_json())
    ids = [x["item_id"] for x in S]
    g = [x["gurmukhi"] for x in S]
    s = [x["shahmukhi"] for x in S]
    cond = {"pan_Guru": g, "pan_Guru_devowel": [devowel_gurmukhi(t) for t in g],
            "pan_Shah": s, "pan_Roman_human": [x["roman"] for x in S]}
    try:
        cond["pan_Guru_rom"] = romanize(g, "pan")
        cond["pan_Shah_rom"] = romanize(s, "pnb")
    except Exception as e:
        print(f"  uroman unavailable ({e}); romanized conditions skipped")
    return ids, cond


PUMVR_CONTRASTS = [
    ("pan_Guru", "pan_Shah", "script + information (human text)", "language, content"),
    ("pan_Guru", "pan_Guru_devowel", "information only", "language, content, script"),
    ("pan_Guru_devowel", "pan_Shah", "script shape (info ~matched)", "language, content"),
    ("pan_Guru_rom", "pan_Shah_rom", "information only, shared Latin", "language, content"),
    ("pan_Guru", "pan_Roman_human", "script (human romanization)", "language, content"),
]


def _load_conditions(args):
    if args.data == "pumvr":
        ids, cond = _pumvr_conditions()
        contrasts = PUMVR_CONTRASTS
    else:
        from conditions import CONTRASTS
        d = load("conditions_flores.json")
        ids, cond, contrasts = d["ids"], d["conditions"], CONTRASTS
    if args.n_sentences and args.n_sentences < len(ids):
        ids = ids[:args.n_sentences]
        cond = {k: v[:args.n_sentences] for k, v in cond.items()}
    return ids, cond, contrasts


# ---------------------------------------------------------------------------
def stage_fertility(args):
    from transformers import AutoTokenizer
    from fertility import fertility_stats, compare_scripts
    from plots import plot_fertility
    ids, cond, _ = _load_conditions(args)

    out, comps, names = {}, [], []
    for name, *_ in selected_models(args):
        print(f"  {name}")
        try:
            tok = AutoTokenizer.from_pretrained(name, cache_dir=args.cache_dir)
        except Exception as e:
            print(f"    SKIP: {type(e).__name__}: {str(e)[:120]}")
            continue
        unk = {tok.unk_token_id} if tok.unk_token_id is not None else None
        per = {c: fertility_stats(t, tok, unk) for c, t in cond.items()}
        cmp_ = compare_scripts(per["pan_Guru"], per["pan_Shah"])
        out[name] = {"per_condition": per, "guru_vs_shah": cmp_}
        comps.append(cmp_)
        names.append(os.path.basename(name.rstrip("/")))
        print(f"    tok/char Shah/Guru={cmp_['tpc']['ratio_b_over_a']:.3f}  "
              f"byte-fallback G={per['pan_Guru']['byte_fallback_rate']:.3f} "
              f"S={per['pan_Shah']['byte_fallback_rate']:.3f}")
    if comps:
        os.makedirs(FIGS, exist_ok=True)
        plot_fertility(comps, f"{FIGS}/fertility_{args.data}.png", names)
    save(f"fertility_{args.data}.json", {"n": len(ids), "per_model": out})


# ---------------------------------------------------------------------------
def stage_audit(args):
    """Sample generated-Shahmukhi pairs for a human check. Checker: the
    independent Shahmukhi->Gurmukhi rules, round-tripped against the source."""
    from data import prioritise_for_audit, write_audit_sheet
    from translit_ctx import ctx_shah_to_gur
    d = load("conditions_flores.json")
    pairs = [{"pair_id": i, "gurmukhi": g, "shahmukhi": s}
             for i, g, s in zip(d["ids"], d["conditions"]["pan_Guru"],
                                d["conditions"]["pan_Shah"])]
    rows, stats = prioritise_for_audit(pairs, ctx_shah_to_gur, n=args.audit_n,
                                       flag_frac=0.6, seed=0)
    path = write_audit_sheet(rows, os.path.join(RESULTS, "audit_sheet.csv"))
    print(f"  flag rate {stats['flag_rate']:.3f} (tau={stats['tau']})  -> {path}")
    print("  Judge the SHAHMUKHI column: is it a correct spelling of the Gurmukhi?")
    save("audit_meta.json", stats)


def stage_score_audit(args):
    from data import score_audit, corpus_error_estimate
    meta = load("audit_meta.json")
    sc = score_audit(os.path.join(RESULTS, "audit_sheet.csv"))
    est = corpus_error_estimate(sc, meta["flag_rate"])
    for k, v in sc["strata"].items():
        print(f"  {k:9s} n={v['n_judged']:4d} err={v['error_rate']:.3f} "
              f"CI [{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}]")
    print(f"  CORPUS ERROR ESTIMATE: {est:.4f}")
    save("audit_scores.json", {**sc, "corpus_error_estimate": est})


# ---------------------------------------------------------------------------
def _extract(texts, name, pool, dt, dmap, args):
    from extract import extract_all_layers
    return extract_all_layers(texts, name, strategy=pool, batch_size=args.batch_size,
                              max_length=args.max_length, dtype=_dtype(dt),
                              device_map=dmap, cache_dir=args.cache_dir,
                              random_init=getattr(args, "random_init", False),
                              seed=getattr(args, "seed", 0))


def _suffix(args):
    if not getattr(args, "random_init", False):
        return ""
    return "_randinit" if args.seed == 0 else f"_randinit_s{args.seed}"


def stage_main(args):
    from analysis import run_contrast, apply_bh, decompose_punjabi_gap, crossover_layer
    from plots import plot_cka_curve, plot_paired_displacement, plot_selectivity

    ids, cond, contrasts = _load_conditions(args)
    n = len(ids)
    print(f"[main] data={args.data} n={n} conditions={len(cond)} contrasts={len(contrasts)}")
    if args.data == "flores" and n < 2000:
        print("  WARNING: n < 2000 (FINDING_01). Power suffers.")
    if args.contrasts:
        want = {x.strip() for x in args.contrasts.split(",")}
        contrasts = [t for t in contrasts if f"{t[0]}|{t[1]}" in want or t[1] in want or t[0] in want]
        if not contrasts:
            raise SystemExit(f"--contrasts matched nothing; available: "
                             f"{[f'{a}|{b}' for a, b, *_ in _load_conditions(args)[2]]}")
        print(f"  contrast subset: {[f'{a}|{b}' for a, b, *_ in contrasts]}")
    needed = sorted({c for a, b, *_ in contrasts for c in (a, b) if c in cond})

    summary = {}
    for name, kind, pool, dt, dmap in selected_models(args):
        tag = os.path.basename(name.rstrip("/"))
        outp = os.path.join(RESULTS, f"main_{args.data}_{tag}{_suffix(args)}.json")
        if os.path.exists(outp) and not args.force:
            print(f"  {tag}: cached")
            continue
        print(f"  {tag}: extracting ({pool} pooling)")
        acts, info = {}, {}
        try:
            for c in needed:
                t0 = time.time()
                cache = (os.path.join(args.acts_dir, f"{args.data}_{tag}{_suffix(args)}_{c}.npy")
                         if args.acts_dir else None)
                if cache and os.path.exists(cache):
                    acts[c] = np.load(cache)
                    info[c] = json.load(open(cache[:-4] + ".json"))
                    src = "cache"
                else:
                    acts[c], info[c] = _extract(cond[c], name, pool, dt, dmap, args)
                    src = "extracted"
                    if cache:
                        os.makedirs(args.acts_dir, exist_ok=True)
                        np.save(cache, acts[c])
                        json.dump(info[c], open(cache[:-4] + ".json", "w"))
                if acts[c].shape[1] != n:
                    raise RuntimeError(f"{c}: cached activations have {acts[c].shape[1]} "
                                       f"items, expected {n}; delete the cache")
                print(f"    {c:18s} {acts[c].shape} trunc={info[c]['truncated_frac']:.3f} "
                      f"{src} ({time.time() - t0:.0f}s)")
        except Exception as e:
            print(f"    SKIP: {type(e).__name__}: {str(e)[:200]}")
            continue
        if args.extract_only:
            print(f"    extract-only: activations saved to {args.acts_dir}")
            del acts
            if args.purge_cache:
                purge_model_cache(name, args.cache_dir)
            continue

        res = {}
        for a, b, varies, fixed in contrasts:
            if a not in acts or b not in acts:
                continue
            key = f"{a}|{b}"
            t0 = time.time()
            res[key] = run_contrast(acts[a], acts[b], ids, C=args.C,
                                    n_perm_cka=args.n_perm, n_perm_pd=args.n_perm,
                                    seed=0, sweep_C=args.sweep_C)
            res[key]["varies"], res[key]["fixed"] = varies, fixed
            print(f"    {key:34s} ({time.time() - t0:.0f}s)")

        n_tests = apply_bh(res)
        dec = decompose_punjabi_gap(res)
        os.makedirs(FIGS, exist_ok=True)
        xo = None
        ftag = tag + _suffix(args)          # figure names only; tag stays clean
        if "pan_Guru|pan_Shah" in res and "pan_Shah|urd_Arab" in res:
            xo = crossover_layer(
                script_sel=[r["selectivity"] for r in res["pan_Guru|pan_Shah"]["probe"]],
                language_sel=[r["selectivity"] for r in res["pan_Shah|urd_Arab"]["probe"]])
            plot_selectivity(res["pan_Guru|pan_Shah"]["probe"],
                             res["pan_Shah|urd_Arab"]["probe"],
                             f"{FIGS}/{args.data}_{ftag}_selectivity.png", ftag, xo)
        if "pan_Guru|pan_Shah" in res:
            plot_cka_curve(res["pan_Guru|pan_Shah"]["cka"],
                           f"{FIGS}/{args.data}_{ftag}_cka.png", ftag, "Gurmukhi vs Shahmukhi")
            plot_paired_displacement(res["pan_Guru|pan_Shah"]["paired"],
                                     f"{FIGS}/{args.data}_{ftag}_paired.png", ftag)

        save(f"main_{args.data}_{tag}{_suffix(args)}.json",
             {"model": name, "kind": kind, "pooling": pool, "n": n,
              "random_init": bool(getattr(args, "random_init", False)),
              "seed": getattr(args, "seed", 0),
              "extraction": info, "bh_family_size": n_tests,
              "crossover_layer": xo, "decomposition": dec, "contrasts": res})
        summary[tag] = {"crossover_layer": xo,
                        "paired_final": {k: v["paired"][-1]["delta"] for k, v in res.items()}}
        del acts
        if args.purge_cache:
            purge_model_cache(name, args.cache_dir)
    if not args.extract_only:
        save(f"main_{args.data}_summary{_suffix(args)}.json", summary)


# ---------------------------------------------------------------------------
def stage_intervention(args):
    from intervention import frozen_transfer, select_layer, summarise_intervention
    sib = load("conditions_sib.json")
    tr, va, te = sib["train"], sib["validation"], sib["test"]
    if args.n_sentences and args.n_sentences < len(tr["label"]):
        tr = {c: v[:args.n_sentences] for c, v in tr.items()}
    test_conds = ["pan_Guru", "pan_Shah", "pan_Shah_to_Guru", "pan_Guru_devowel"]
    y_tr, y_va, y_te = np.array(tr["label"]), np.array(va["label"]), np.array(te["label"])

    out = {}
    for name, kind, pool, dt, dmap in selected_models(args):
        tag = os.path.basename(name.rstrip("/"))
        print(f"  {tag}")
        try:
            def ex(texts):
                return _extract(texts, name, pool, dt, dmap, args)[0].astype(np.float32)
            A_tr, A_va = ex(tr["pan_Guru"]), ex(va["pan_Guru"])
            A_te = {c: ex(te[c]) for c in test_conds}
            A_tr_s = ex(tr["pan_Shah"])
        except Exception as e:
            print(f"    SKIP: {type(e).__name__}: {str(e)[:200]}")
            continue

        best, val_curve = select_layer(A_tr, y_tr, A_va, y_va, C=args.C)
        res = frozen_transfer(A_tr[best], y_tr,
                              {c: (A_te[c][best], y_te) for c in test_conds}, C=args.C)
        summ = summarise_intervention(res)
        ins = frozen_transfer(A_tr_s[best], y_tr, {"s": (A_te["pan_Shah"][best], y_te)}, C=args.C)
        summ["shah_trained_in_script_acc"] = ins["s"]["acc"]

        curve = []
        for li in range(A_tr.shape[0]):
            r = frozen_transfer(A_tr[li], y_tr, {c: (A_te[c][li], y_te) for c in test_conds},
                                C=args.C)
            curve.append({"layer": li, **{c: r[c]["acc"] for c in test_conds}})

        maj = float(max(np.mean(y_te == c) for c in set(y_te.tolist())))
        out[tag] = {"best_layer": best, "val_curve": val_curve, "summary": summ,
                    "layer_curve": curve, "majority": maj}
        a = summ["acc"]
        print(f"    L{best}: Guru={a['pan_Guru']:.3f} Shah={a['pan_Shah']:.3f} "
              f"restored={a['pan_Shah_to_Guru']:.3f} devowel={a['pan_Guru_devowel']:.3f} "
              f"gap closed={summ['gap_closed_frac']:.2f}  (majority {maj:.3f})")
    save("intervention_sib.json", out)


def stage_intervention_mcq(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from intervention import mcq_accuracy, mcnemar_exact, paired_bootstrap_diff
    from translit_ctx import ctx_shah_to_gur
    items = json.load(open(pumvr_json(), encoding="utf-8"))
    if args.n_sentences and args.n_sentences < len(items):
        items = items[:args.n_sentences]

    out = {}
    for name, kind, pool, dt, dmap in selected_models(args):
        if kind != "decoder":
            continue
        tag = os.path.basename(name.rstrip("/"))
        print(f"  {tag}")
        try:
            tok = AutoTokenizer.from_pretrained(name, cache_dir=args.cache_dir)
            kw = dict(dtype=_dtype(dt), cache_dir=args.cache_dir)
            if dmap:
                model = AutoModelForCausalLM.from_pretrained(name, device_map=dmap, **kw).eval()
                dev = next(model.parameters()).device
            else:
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                model = AutoModelForCausalLM.from_pretrained(name, **kw).to(dev).eval()
        except Exception as e:
            print(f"    SKIP: {type(e).__name__}: {str(e)[:200]}")
            continue
        c = {"gurmukhi": mcq_accuracy(model, tok, items, "gurmukhi", dev),
             "shahmukhi": mcq_accuracy(model, tok, items, "shahmukhi", dev),
             "shahmukhi_to_gurmukhi": mcq_accuracy(model, tok, items, "shahmukhi", dev,
                                                   transform=ctx_shah_to_gur),
             "roman": mcq_accuracy(model, tok, items, "roman", dev)}
        rec = {"acc": {k: float(v.mean()) for k, v in c.items()}, "n": len(items),
               "gain": paired_bootstrap_diff(c["shahmukhi"], c["shahmukhi_to_gurmukhi"]),
               "mcnemar_shah_vs_restored": mcnemar_exact(c["shahmukhi"],
                                                         c["shahmukhi_to_gurmukhi"]),
               "mcnemar_guru_vs_shah": mcnemar_exact(c["gurmukhi"], c["shahmukhi"])}
        out[tag] = rec
        print(f"    {rec['acc']}  gain={rec['gain'][0]:+.3f}")
        del model
    save("intervention_mcq.json", out)


# ---------------------------------------------------------------------------
def stage_report(args):
    """Every JSON in results/ condensed into the tables the paper needs."""
    L = ["# MUKHI results report", "",
         "Generated by `run_all.py --stage report`. Numbers only; interpretation "
         "lives in the paper draft.", ""]

    def have(n):
        return os.path.exists(os.path.join(RESULTS, n))

    if have("translit_eval.json"):
        t = load("translit_eval.json")
        L += ["## Transliteration (PuMVR, held-out)", "",
              "| system | test-word CER | sentence CER |", "|---|---:|---:|",
              f"| S->G naive map | {t['s2g_naive_test_words']['cer']:.3f} | "
              f"{t['s2g_naive_sentences']['cer']:.3f} |",
              f"| S->G context rules | {t['s2g_context_test_words']['cer']:.3f} | "
              f"{t['s2g_context_sentences']['cer']:.3f} |",
              f"| G->S rules | {t['g2s_rules_test_words_cer']:.3f} | "
              f"{t['g2s_rules_sentences_cer']:.3f} |", "",
              f"Unwritten-vowel share of the S->G residual: "
              f"{t['s2g_context_unwritten_share']:.3f}", ""]

    for data in ("flores", "pumvr"):
        if have(f"fertility_{data}.json"):
            f = load(f"fertility_{data}.json")
            L += [f"## Tokeniser premium ({data})", "",
                  "| model | tok/char Shah / Guru | Wilcoxon p | byte-fallback Guru | "
                  "byte-fallback Shah |", "|---|---:|---:|---:|---:|"]
            for m, v in f["per_model"].items():
                c = v["guru_vs_shah"]["tpc"]
                pc = v["per_condition"]
                L.append(f"| {os.path.basename(m.rstrip('/'))} | {c['ratio_b_over_a']:.3f} | "
                         f"{c['wilcoxon_p']:.1e} | {pc['pan_Guru']['byte_fallback_rate']:.3f} | "
                         f"{pc['pan_Shah']['byte_fallback_rate']:.3f} |")
            L.append("")

        files = [p for p in sorted(glob.glob(os.path.join(RESULTS, f"main_{data}_*.json")))
                 if "_summary" not in p and "_randinit" not in p]
        if files:
            L += [f"## Representation contrasts ({data})", "",
                  "Paired-displacement delta (content cancelled) at the last layer for the "
                  "trained model and for the SAME architecture with random weights (mean over "
                  "seeds run, with the seed range). A trained-minus-random difference smaller than "
                  "the seed range is noise. Only the "
                  "difference is evidence of anything learned. Sig. = layers significant after "
                  "BH over the whole family (trained model).", "",
                  "| model | contrast | paired Llast trained | paired Llast random-init | "
                  "trained minus random | CKA delta max | sig. layers |",
                  "|---|---|---:|---:|---:|---:|---:|"]
            for p in files:
                d = json.load(open(p, encoding="utf-8"))
                tag = os.path.basename(d["model"].rstrip("/"))
                seeds = [json.load(open(q, encoding="utf-8"))["contrasts"]
                         for q in sorted(glob.glob(p[:-5] + "_randinit*.json"))]
                for k, v in d["contrasts"].items():
                    pdl = v["paired"]
                    sig = sum(1 for r in pdl if r.get("significant_bh"))
                    last = pdl[-1]["delta"]
                    rv = [sd[k]["paired"][-1]["delta"] for sd in seeds if k in sd]
                    if rv:
                        rm = sum(rv) / len(rv)
                        rl = f"{rm:.3f}" + (f" [{min(rv):.3f}, {max(rv):.3f}] n={len(rv)}"
                                            if len(rv) > 1 else " (1 seed)")
                        df = f"{last - rm:+.3f}"
                    else:
                        rl, df = "not run", "n/a"
                    L.append(f"| {tag} | {k.replace('|', ' vs ')} | {last:.3f} | {rl} | {df} | "
                             f"{max(r['delta'] for r in v['cka']):.3f} | {sig}/{len(pdl)} |")
            L += ["", "### Crossover and information / shape split", "",
                  "| model | crossover layer | mean shape share of the Guru-Shah gap |",
                  "|---|---:|---:|"]
            for p in files:
                d = json.load(open(p, encoding="utf-8"))
                tag = os.path.basename(d["model"].rstrip("/"))
                ss = [r["shape_share"] for r in (d.get("decomposition") or [])
                      if r["shape_share"] == r["shape_share"]]
                share = f"{sum(ss) / len(ss):.3f}" if ss else "n/a"
                L.append(f"| {tag} | {d.get('crossover_layer')} | {share} |")
            L.append("")

    if have("intervention_sib.json"):
        iv = load("intervention_sib.json")
        L += ["## Intervention: SIB-200 topic classification, trained on Gurmukhi", "",
              "| model | layer | Guru | Shah | Shah->Guru | Guru devowel | Shah in-script | "
              "gap closed | McNemar p |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for m, v in iv.items():
            s, a = v["summary"], v["summary"]["acc"]
            L.append(f"| {m} | {v['best_layer']} | {a['pan_Guru']:.3f} | {a['pan_Shah']:.3f} | "
                     f"{a['pan_Shah_to_Guru']:.3f} | {a['pan_Guru_devowel']:.3f} | "
                     f"{s['shah_trained_in_script_acc']:.3f} | "
                     f"{'n/a' if s['gap_closed_frac'] != s['gap_closed_frac'] else format(s['gap_closed_frac'], '.2f')} | "
                     f"{s['mcnemar_shah_vs_restored'][2]:.1e} |")
        L.append("")

    if have("intervention_mcq.json"):
        mq = load("intervention_mcq.json")
        L += ["## Intervention: PuMVR text-only option scoring, human text", "",
              "| model | Guru | Shah | Shah->Guru | Roman | gain | McNemar p |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for m, v in mq.items():
            a = v["acc"]
            L.append(f"| {m} | {a['gurmukhi']:.3f} | {a['shahmukhi']:.3f} | "
                     f"{a['shahmukhi_to_gurmukhi']:.3f} | {a['roman']:.3f} | "
                     f"{v['gain'][0]:+.3f} | {v['mcnemar_shah_vs_restored'][2]:.1e} |")
        L.append("")

    path = os.path.join(RESULTS, "REPORT.md")
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
STAGES = {"smoke": stage_smoke, "translit_eval": stage_translit_eval,
          "g2s_compare": stage_g2s_compare, "conditions": stage_conditions,
          "fertility": stage_fertility, "audit": stage_audit,
          "score_audit": stage_score_audit, "main": stage_main,
          "intervention": stage_intervention, "intervention_mcq": stage_intervention_mcq,
          "report": stage_report}


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    ap.add_argument("--data", default="flores", choices=["flores", "pumvr"])
    ap.add_argument("--models", default="", help="comma-separated HF ids or local dirs")
    ap.add_argument("--n-sentences", type=int, default=0, dest="n_sentences",
                    help="0 = all; smaller values for quick checks")
    ap.add_argument("--audit-n", type=int, default=300, dest="audit_n")
    ap.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    ap.add_argument("--max-length", type=int, default=512, dest="max_length")
    ap.add_argument("--n-perm", type=int, default=200, dest="n_perm")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--sweep-C", action="store_true", dest="sweep_C")
    ap.add_argument("--cache-dir", default=None, dest="cache_dir")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--random-init", action="store_true", dest="random_init",
                    help="same architecture, random weights: the baseline for main")
    ap.add_argument("--purge-cache", action="store_true", dest="purge_cache",
                    help="delete each model's weights after use; Kaggle's 20GB "
                         "working disk fills after ~3 large models")
    ap.add_argument("--contrasts", default="", help="comma-separated subset, "
                    "e.g. 'pan_Guru|pan_Shah,srp_Cyrl|srp_Latn'; default all")
    ap.add_argument("--acts-dir", default=None, dest="acts_dir",
                    help="save/load activations here (fp16 .npy per model x condition)")
    ap.add_argument("--extract-only", action="store_true", dest="extract_only",
                    help="GPU session: extract and save, skip analysis (needs --acts-dir)")
    ap.add_argument("--seed", type=int, default=0,
                    help="random-init seed; run 3+ seeds and the report averages them")
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    if args.extract_only and not args.acts_dir:
        ap.error("--extract-only needs --acts-dir, or the extraction is thrown away")
    STAGES[args.stage](args)


if __name__ == "__main__":
    main()
