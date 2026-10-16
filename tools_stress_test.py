"""Stress test of the paired statistics on synthetic data with a planted
script displacement of fixed size. Each sweep changes one nuisance and holds
the script signal constant, so any movement in a statistic is a failure of
that statistic. Writes results/stress_test.json.

Nuisances:
  scale_b      rendering B multiplied by k (norm mismatch between scripts)
  item_scale   each sentence's two renderings scaled by its own random factor
  content_norm content variance grown with the script signal fixed
  anisotropy   content variance concentrated in few directions
  alignment    script direction moved into the dominant content direction
Interactions (content and script not additive):
  gain         script displacement grows with each sentence's content
  compress     the second script shrinks the shared content as well as adding
               its own displacement
  tilt         each sentence's script direction leans toward its own content
  depth        content variance grows and the script direction rotates
               together, layer by layer, as in a real stack
Reference: the same statistics when the planted script signal is halved.
"""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from bootstrap import per_item_deltas, point_stats  # noqa: E402

N, D = 600, 64


def make(rng, content_sd=1.0, script=1.0, cond=1.0, align=0.0, scale_b=1.0,
         item_scale=0.0, noise=0.3, gain=0.0, compress=0.0, tilt=0.0):
    ev = np.geomspace(cond, 1.0, D)            # anisotropy: eigen spread
    ev = ev / ev.mean()
    C = rng.normal(size=(N, D)) * np.sqrt(ev) * content_sd
    u = np.zeros(D); u[0] = 1.0                # dominant content direction
    v = np.zeros(D); v[-1] = 1.0               # weakest content direction
    s = np.sqrt(align) * u + np.sqrt(1 - align) * v
    A = C + noise * rng.normal(size=(N, D))
    S = np.tile(s, (N, 1))
    if tilt:
        own = C / np.linalg.norm(C, axis=1, keepdims=True)
        S = S + tilt * own
        S = S / np.linalg.norm(S, axis=1, keepdims=True)
    if gain:
        S = S * (1.0 + gain * np.abs(C @ u)[:, None] / content_sd)
    B = (1.0 - compress) * C + script * S + noise * rng.normal(size=(N, D))
    B = B * scale_b
    if item_scale:
        f = np.exp(item_scale * rng.normal(size=(N, 1)))
        A, B = A * f, B * f * np.exp(item_scale * rng.normal(size=(N, 1)))
    return point_stats(*per_item_deltas(A, B))


def sweep(name, key, values, **fixed):
    rows = []
    for v in values:
        rng = np.random.default_rng(0)
        c, sh, rm = make(rng, **{key: v}, **fixed)
        rows.append({"value": v, "consistency": c, "shared": sh, "rms": rm})
    return {"name": name, "param": key, "rows": rows}


def depth():
    """A synthetic stack: content variance grows 1 to 8 and the script
    direction rotates from the weakest to the dominant content direction,
    while the planted script size stays fixed."""
    rows = []
    for i, t in enumerate(np.linspace(0, 1, 9)):
        rng = np.random.default_rng(0)
        c, sh, rm = make(rng, content_sd=float(2 ** (3 * t)), align=float(t))
        rows.append({"value": i, "consistency": c, "shared": sh, "rms": rm})
    return {"name": "depth: variance growth and rotation together", "param": "layer",
            "rows": rows}


def main():
    base = make(np.random.default_rng(0))
    half = make(np.random.default_rng(0), script=0.5)
    ref = {k: abs(a - b) for k, a, b in zip(("consistency", "shared", "rms"), base, half)}
    sweeps = [
        sweep("norm mismatch between scripts", "scale_b", [0.25, 0.5, 1, 2, 4]),
        sweep("per-sentence scale", "item_scale", [0.0, 0.25, 0.5, 1.0]),
        sweep("content variance growth", "content_sd", [0.5, 1, 2, 4, 8]),
        sweep("anisotropy (eigenvalue spread)", "cond", [1, 10, 100, 1000]),
        sweep("script aligned with dominant content", "align", [0.0, 0.25, 0.5, 0.75, 1.0]),
        sweep("interaction: script size grows with content", "gain", [0.0, 0.5, 1.0, 2.0]),
        sweep("interaction: second script compresses content", "compress", [0.0, 0.1, 0.25, 0.5]),
        sweep("interaction: script direction tilts to content", "tilt", [0.0, 0.25, 0.5, 1.0]),
        depth(),
    ]
    for s in sweeps:
        for k in ("consistency", "shared", "rms"):
            vals = [r[k] for r in s["rows"]]
            s[k + "_range"] = max(vals) - min(vals)
            s[k + "_vs_halving"] = s[k + "_range"] / ref[k]
    out = {"n": N, "d": D, "baseline": dict(zip(("consistency", "shared", "rms"), base)),
           "halving_effect": ref, "sweeps": sweeps,
           "reading": "x_vs_halving is the nuisance-induced range divided by the "
                      "change caused by halving the real script signal. Below "
                      "0.1 the statistic is stable under that nuisance; above "
                      "1 the nuisance moves it more than the signal does."}
    json.dump(out, open(os.path.join(ROOT, "results", "stress_test.json"), "w"), indent=2)
    print("halving the script signal moves: " +
          ", ".join("%s %.3f" % kv for kv in ref.items()))
    print("%-40s %12s %8s %8s" % ("nuisance", "consistency", "shared", "rms"))
    for s in sweeps:
        print("%-40s %12.2f %8.2f %8.2f" % (s["name"], s["consistency_vs_halving"],
                                           s["shared_vs_halving"], s["rms_vs_halving"]))


if __name__ == "__main__":
    main()
