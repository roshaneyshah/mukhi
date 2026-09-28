import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from fertility import (count_words, count_chars, detect_script,
                       fertility_stats, compare_scripts)

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("PASS  " if cond else "FAIL  ") + name + ("   " + detail if detail else ""))

# --- script detection on real codepoints -----------------------------------
guru = "ਪੰਜਾਬੀ"          # Gurmukhi letters
shah = "پنجابی"          # Shahmukhi letters
check("detects gurmukhi", detect_script(guru) == "gurmukhi", detect_script(guru))
check("detects shahmukhi", detect_script(shah) == "shahmukhi", detect_script(shah))
check("detects latin", detect_script("panjabi") == "latin")
check("detects mixed", detect_script(guru + shah) == "mixed", detect_script(guru+shah))

# --- zero-width joiners must not create words or count as chars ------------
with_zw = "پن‌ج اب"
check("ZWNJ does not split words", count_words(with_zw) == 2, str(count_words(with_zw)))
check("ZWNJ excluded from char count", count_chars(with_zw) == 5, str(count_chars(with_zw)))
check("whitespace excluded from char count", count_chars("ab  cd\n") == 4)

# --- fertility arithmetic against a mock tokenizer -------------------------
class MockTok:
    """1 token per char, plus a byte-fallback piece for every Arabic-range char."""
    def encode(self, t):
        ids = []
        for ch in t.replace(" ", ""):
            ids.append(ord(ch))
            if 0x0600 <= ord(ch) <= 0x06FF:
                ids.append(-1)              # simulate byte fallback
        return ids
    def convert_ids_to_tokens(self, ids):
        return ["<0xE0>" if i == -1 else chr(i) for i in ids]

tok = MockTok()
g_texts = ["ਪੰ ਜਾ"]     # 4 chars, 2 words -> 4 tokens
s_texts = ["پن جا"]     # 4 chars, 2 words -> 8 tokens (fallback)

gs = fertility_stats(g_texts, tok)
ss = fertility_stats(s_texts, tok)
check("mock: gurmukhi tokens_per_char == 1.0", abs(gs["tokens_per_char_mean"]-1.0)<1e-9,
      f"{gs['tokens_per_char_mean']:.3f}")
check("mock: shahmukhi tokens_per_char == 2.0", abs(ss["tokens_per_char_mean"]-2.0)<1e-9,
      f"{ss['tokens_per_char_mean']:.3f}")
check("mock: gurmukhi byte_fallback_rate == 0", gs["byte_fallback_rate"] == 0.0)
check("mock: shahmukhi byte_fallback_rate == 0.5", abs(ss["byte_fallback_rate"]-0.5)<1e-9,
      f"{ss['byte_fallback_rate']:.3f}")
check("mock: tokens_per_word == 2 for gurmukhi", abs(gs["tokens_per_word_mean"]-2.0)<1e-9)

# --- byte-level BPE fallback must be caught by fragment_rate ----------------
# byte_fallback_rate only sees SentencePiece "<0x..>" pieces, so Qwen/Aya-style
# byte-level BPE scored 0.000 on FLORES Gurmukhi while every token was a byte.
try:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast
    _tk = Tokenizer(models.BPE())
    _tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    _tk.decoder = decoders.ByteLevel()
    _tk.train_from_iterator(["hello world the quick brown fox"] * 200,
        trainers.BpeTrainer(vocab_size=300,
                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    _fast = PreTrainedTokenizerFast(tokenizer_object=_tk)
    _g = fertility_stats(["\u0a2a\u0a70\u0a1c\u0a3e\u0a2c\u0a40"], _fast)
    _l = fertility_stats(["hello world the quick brown fox"], _fast)
    check("byte-level BPE: byte_fallback_rate misses it (documented)",
          _g["byte_fallback_rate"] == 0.0)
    check("byte-level BPE: fragment_rate catches it", _g["fragment_rate"] > 0.8,
          f"{_g['fragment_rate']:.3f}")
    check("byte-level BPE: in-vocabulary script has fragment_rate 0",
          _l["fragment_rate"] == 0.0, f"{_l['fragment_rate']:.3f}")
except ImportError as _e:
    print("SKIP  byte-level BPE check:", _e)

# --- paired comparison ------------------------------------------------------
rng = np.random.default_rng(0)
gg = ["ਪ"*int(n) for n in rng.integers(5, 20, 60)]
ss2 = [t.replace("ਪ", "پ") for t in gg]     # same lengths, other script
sa, sb = fertility_stats(gg, tok), fertility_stats(ss2, tok)
cmp = compare_scripts(sa, sb)
check("paired: ratio ~2.0 on tokens", abs(cmp["tokens"]["ratio_b_over_a"]-2.0)<1e-6,
      f"{cmp['tokens']['ratio_b_over_a']:.4f}")
check("paired: wilcoxon p significant", cmp["tokens"]["wilcoxon_p"] < 1e-6,
      f"p={cmp['tokens']['wilcoxon_p']:.2e}")
check("paired: n_pairs correct", cmp["tokens"]["n_pairs"] == 60)

# --- identical corpora must NOT come out significant -----------------------
cmp_null = compare_scripts(sa, sa)
check("paired: identical corpora ratio == 1.0",
      abs(cmp_null["tokens"]["ratio_b_over_a"]-1.0)<1e-12)

# --- empty / degenerate input ----------------------------------------------
es = fertility_stats([""], tok)
check("empty string does not crash", np.isnan(es["tokens_per_word_mean"]) or es["tokens_per_word_mean"]==0,
      str(es["tokens_per_word_mean"]))

print()
nf = sum(1 for _,ok,_ in results if not ok)
print(f"{len(results)-nf}/{len(results)} passed")
sys.exit(1 if nf else 0)
