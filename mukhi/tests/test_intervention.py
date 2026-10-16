import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from intervention import (mcnemar_exact, paired_bootstrap_diff, frozen_transfer,
                          select_layer, summarise_intervention)

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("PASS  " if cond else "FAIL  ") + name + ("   " + detail if detail else ""))

# --- McNemar -----------------------------------------------------------------
a = np.array([1]*40 + [0]*10 + [1]*45 + [0]*5, bool)
b = np.array([1]*40 + [1]*10 + [0]*45 + [0]*5, bool)   # 45 vs 10 discordant
n01, n10, p = mcnemar_exact(a, b)
check("mcnemar counts discordant pairs", (n01, n10) == (45, 10), str((n01, n10)))
check("mcnemar significant for 45 vs 10", p < 1e-4, f"p={p:.2e}")
check("mcnemar p=1 for identical", mcnemar_exact(a, a)[2] == 1.0)
d, lo, hi = paired_bootstrap_diff(a, b)
check("paired diff sign and CI", d < 0 and hi < 0, f"{d:.3f} [{lo:.3f},{hi:.3f}]")

# --- controlled arm on planted features -------------------------------------
# Label lives in direction w. Gurmukhi features: w fully visible.
# Shahmukhi: label signal partly scrambled (script bottleneck).
# Restored: scramble removed but a slice of the signal deleted (lost info).
# Devowel: same slice deleted, no scramble.  Expect restored ~= devowel < guru.
rng = np.random.default_rng(0)
n, d, k = 600, 40, 7
y = rng.integers(0, k, n)
proto = np.zeros((k, d))
proto[:, :16] = rng.normal(size=(k, 16)) * 1.1     # label signal only in dims 0-15
base = proto[y] + rng.normal(size=(n, d))
lost = np.zeros(d); lost[:12] = 1                         # dims the input "never had"
R = np.linalg.qr(rng.normal(size=(d, d)))[0]
guru = base
devow = base * (1 - lost) + rng.normal(size=(n, d)) * lost
shah = devow @ R * 0.6 + rng.normal(size=(n, d)) * 0.8
rest = devow + rng.normal(size=(n, d)) * 0.05
tr, te = np.arange(0, 400), np.arange(400, 600)
res = frozen_transfer(guru[tr], y[tr], {
    "pan_Guru": (guru[te], y[te]), "pan_Shah": (shah[te], y[te]),
    "pan_Shah_to_Guru": (rest[te], y[te]), "pan_Guru_devowel": (devow[te], y[te])})
summ = summarise_intervention(res)
acc = summ["acc"]
check("planted: guru > restored", acc["pan_Guru"] > acc["pan_Shah_to_Guru"] + 0.05, str(acc))
check("planted: restored > shah (intervention helps)", summ["gain_restored_minus_shah"] > 0.1,
      f"gain={summ['gain_restored_minus_shah']:.3f} CI={summ['gain_ci95']}")
check("planted: restored ~= devowel (residual gap is lost information)",
      abs(acc["pan_Shah_to_Guru"] - acc["pan_Guru_devowel"]) < 0.06
      and summ["mcnemar_restored_vs_devowel"][2] > 0.05,
      f"{acc['pan_Shah_to_Guru']:.3f} vs {acc['pan_Guru_devowel']:.3f} p={summ['mcnemar_restored_vs_devowel'][2]:.2f}")
check("planted: gap_closed_frac in (0,1)", 0 < summ["gap_closed_frac"] < 1, f"{summ['gap_closed_frac']:.3f}")

# --- gap fraction is undefined without a significant positive gap ------------
flat = {"pan_Guru": {"acc": .16, "correct": np.array([1,0,0,0,0,0,0,0,0,0]*14, bool)},
        "pan_Shah": {"acc": .18, "correct": np.array([1,1,0,0,0,0,0,0,0,0]*14, bool)[::-1]},
        "pan_Shah_to_Guru": {"acc": .14, "correct": np.array([0,1,0,0,0,0,0,0,0,0]*14, bool)},
        "pan_Guru_devowel": {"acc": .13, "correct": np.array([0,0,1,0,0,0,0,0,0,0]*14, bool)}}
sf = summarise_intervention(flat)
check("gap_closed_frac is NaN when Shah >= Guru (dry-run bug guarded)",
      sf["gap_closed_frac"] != sf["gap_closed_frac"], f"{sf['gap_closed_frac']} / {sf['gap_closed_note']}")
check("gap_closed_frac defined on a real gap", summ["gap_closed_note"] == "ok")

# --- layer selection uses the ceiling condition -------------------------------
acts_tr = np.stack([rng.normal(size=(400, d)), guru[tr]])      # layer 1 informative
acts_va = np.stack([rng.normal(size=(200, d)), guru[te]])
best, scores = select_layer(acts_tr, y[tr], acts_va, y[te])
check("select_layer picks the informative layer", best == 1, str(scores))

# --- LM option scoring through real transformers path -------------------------
try:
    import torch
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, GPT2Config, GPT2LMHeadModel
    from intervention import score_options, mcq_accuracy
    corpus = ["ਪੱਗ ਸਿਰ ਜੁੱਤੀ ਪੈਰ ਹੱਥ ਅੱਖ ਨੱਕ", "پگ سر جُتی پیر ہتھ اکھ نک"] * 30
    tk = Tokenizer(models.WordPiece(unk_token="[UNK]")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.train_from_iterator(corpus, trainers.WordPieceTrainer(vocab_size=200, special_tokens=["[PAD]","[UNK]","[BOS]"]))
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]", unk_token="[UNK]", bos_token="[BOS]")
    torch.manual_seed(0)
    m = GPT2LMHeadModel(GPT2Config(vocab_size=fast.vocab_size, n_embd=32, n_layer=2, n_head=2)).eval()
    sc = score_options(m, fast, "ਪੱਗ : ਸਿਰ", ["ਹੱਥ", "ਪੈਰ", "ਅੱਖ", "ਨੱਕ"])
    check("score_options returns one finite score per option",
          sc.shape == (4,) and np.all(np.isfinite(sc)), str(np.round(sc, 3)))
    item = {"scripts": {"gurmukhi": {"question": "ਪੱਗ : ਸਿਰ", "options": ["ਹੱਥ", "ਪੈਰ", "ਅੱਖ", "ਨੱਕ"], "answer": "ਪੈਰ"},
                        "shahmukhi": {"question": "پگ : سر", "options": ["ہتھ", "پیر", "اکھ", "نک"], "answer": "پیر"}}}
    cg = mcq_accuracy(m, fast, [item] * 3, "gurmukhi")
    check("mcq_accuracy returns a boolean vector per item", cg.dtype == bool and len(cg) == 3)
    from translit_ctx import ctx_shah_to_gur
    cs = mcq_accuracy(m, fast, [item], "shahmukhi", transform=ctx_shah_to_gur)
    check("mcq_accuracy accepts an intervention transform", len(cs) == 1)
except ImportError as e:
    print("SKIP  LM scoring:", e)

print()
nf = sum(1 for _, ok, _ in results if not ok)
print(f"{len(results)-nf}/{len(results)} passed")
sys.exit(1 if nf else 0)
