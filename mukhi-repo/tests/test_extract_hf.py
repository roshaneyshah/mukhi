"""
Runs extract_all_layers through the real transformers code path using tiny,
randomly initialised models and a tokenizer trained locally. No network.
Checks shapes, layer count, truncation accounting, pooling equivalence with a
manual forward pass, and padding-side invariance for a decoder.
"""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("PASS  " if cond else "FAIL  ") + name + ("   " + detail if detail else ""))

try:
    import torch
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    from transformers import (PreTrainedTokenizerFast, BertConfig, BertModel,
                              GPT2Config, GPT2Model)
except Exception as e:
    print("SKIP  torch/transformers unavailable:", e); sys.exit(0)

from extract import extract_all_layers, pool_hidden

texts = ["ਪੰਜਾਬੀ ਭਾਸ਼ਾ ਬਹੁਤ ਸੋਹਣੀ ਹੈ", "پنجابی زبان", "ਹੱਥ", "ਇੱਕ ਦੋ ਤਿੰਨ ਚਾਰ ਪੰਜ ਛੇ ਸੱਤ ਅੱਠ ਨੌਂ ਦਸ " * 6,
         "اردو", "Dobar dan", "ਕੁਰਸੀ ਮੇਜ਼"]

tmp = tempfile.mkdtemp()
tk = Tokenizer(models.WordPiece(unk_token="[UNK]"))
tk.pre_tokenizer = pre_tokenizers.Whitespace()
tk.train_from_iterator(texts * 20, trainers.WordPieceTrainer(
    vocab_size=300, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]))
fast = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]", unk_token="[UNK]",
                               cls_token="[CLS]", sep_token="[SEP]", eos_token="[SEP]")

torch.manual_seed(0)
enc_dir = os.path.join(tmp, "enc")
BertModel(BertConfig(vocab_size=fast.vocab_size, hidden_size=32, num_hidden_layers=3,
                     num_attention_heads=2, intermediate_size=64,
                     max_position_embeddings=512)).save_pretrained(enc_dir)
fast.save_pretrained(enc_dir)

acts, info = extract_all_layers(texts, enc_dir, strategy="mean", batch_size=3,
                                max_length=16, device="cpu")
check("encoder: shape (layers+1, n, d)", acts.shape == (4, len(texts), 32), str(acts.shape))
check("encoder: truncation counted (the long text)", info["truncated"] == 1, str(info))
check("encoder: all finite", np.all(np.isfinite(acts.astype(np.float32))))

# batch-size invariance: pooling must not depend on how texts were batched
acts1, _ = extract_all_layers(texts, enc_dir, strategy="mean", batch_size=1,
                              max_length=16, device="cpu")
diff = float(np.max(np.abs(acts.astype(np.float32) - acts1.astype(np.float32))))
check("encoder: batch-size invariant (padding masked correctly)", diff < 5e-3, f"max diff {diff:.2e}")

# decoder, padding-side invariance
dec_dir = os.path.join(tmp, "dec")
GPT2Model(GPT2Config(vocab_size=fast.vocab_size, n_embd=32, n_layer=2, n_head=2,
                     n_positions=512)).save_pretrained(dec_dir)
fast.save_pretrained(dec_dir)
d_b, dinfo = extract_all_layers(texts, dec_dir, strategy="last", batch_size=4,
                                max_length=64, device="cpu")
d_1, _ = extract_all_layers(texts, dec_dir, strategy="last", batch_size=1,
                            max_length=64, device="cpu")
check("decoder: shape", d_b.shape == (3, len(texts), 32), str(d_b.shape))
dd = float(np.max(np.abs(d_b.astype(np.float32) - d_1.astype(np.float32))))
check("decoder: last-token pooling batch-size invariant", dd < 5e-3, f"max diff {dd:.2e}")

# pooling matches a manual forward pass on one text
m = GPT2Model.from_pretrained(dec_dir).eval()
ids = fast(texts[0], return_tensors="pt")
with torch.no_grad():
    hs = m(**ids, output_hidden_states=True).hidden_states
manual = hs[-1][0, -1].numpy()
dm = float(np.max(np.abs(manual - d_1[-1, 0].astype(np.float32))))
check("decoder: matches manual last-token hidden state", dm < 5e-3, f"max diff {dm:.2e}")

# random-init baseline: same shapes, different weights across seeds, deterministic per seed
r0, _ = extract_all_layers(texts, enc_dir, strategy="mean", batch_size=4, max_length=16,
                           device="cpu", random_init=True, seed=1)
r0b, _ = extract_all_layers(texts, enc_dir, strategy="mean", batch_size=4, max_length=16,
                            device="cpu", random_init=True, seed=1)
r1, _ = extract_all_layers(texts, enc_dir, strategy="mean", batch_size=4, max_length=16,
                           device="cpu", random_init=True, seed=2)
check("random_init: same shape as trained", r0.shape == acts.shape, str(r0.shape))
check("random_init: deterministic per seed", np.array_equal(r0, r0b))
check("random_init: differs across seeds", not np.allclose(r0.astype(np.float32), r1.astype(np.float32)))

print()
nf = sum(1 for _, ok, _ in results if not ok)
print(f"{len(results)-nf}/{len(results)} passed")
sys.exit(1 if nf else 0)
