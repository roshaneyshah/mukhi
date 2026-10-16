"""
Tokenizer fertility across scripts.

Proposal step 4 asks whether the tokenizer explains the whole script effect.
Fertility alone does not answer that, so this computes four quantities:

  tokens_per_word  - standard fertility. Sensitive to how you count words,
                     which is NOT comparable across scripts by default (see
                     `count_words`).
  tokens_per_char  - script-robust. Use this for the cross-script claim;
                     word counts differ between Gurmukhi and Shahmukhi for
                     orthographic reasons unrelated to the tokenizer.
  tokens_per_sent  - the tokenization premium a user actually pays.
  unk_rate / byte_fallback_rate / fragment_rate - the sharpest test. If one
                     script is largely byte fragments, the model is not "worse"
                     at it, it is barely reading it, and every downstream
                     representational claim inherits that.

                     byte_fallback_rate only catches SentencePiece "<0x..>"
                     pieces. Byte-level BPE (Qwen, Aya, GPT-2 style) encodes
                     raw bytes as ordinary visible tokens, so it scores 0.000
                     there while every token is a byte. fragment_rate catches
                     both: it decodes each token ALONE and counts those that
                     do not form a whole character (U+FFFD or empty). Verified
                     against a byte-level tokenizer trained without Gurmukhi:
                     byte_fallback_rate 0.000, fragment_rate 1.000.

Note the direction is an open question, not an assumption. PuMVR (2026)
reports Gurmukhi as the brittle script in some few-shot settings, and recent
pre-tokenization work argues abugida scripts (Gurmukhi) carry the fertility
penalty, not Perso-Arabic. Measure it.
"""
import re
import warnings
import numpy as np

# Unicode ranges
GURMUKHI = (0x0A00, 0x0A7F)
ARABIC_BLOCKS = [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
                 (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]

# Zero-width joiner/non-joiner: present in Shahmukhi, absent in Gurmukhi.
# They are not word boundaries and must not be counted as such.
ZW = "‌‍"


def count_words(text, script=None):
    """
    Whitespace word count, with zero-width joiners stripped first.

    Comparing raw word counts ACROSS scripts is not meaningful on its own:
    Shahmukhi writes some compounds solid that Gurmukhi spaces, so identical
    content yields different word counts. Use tokens_per_char for cross-script
    claims and tokens_per_word only within a script.
    """
    t = text
    for ch in ZW:
        t = t.replace(ch, "")
    return len([w for w in t.split() if w.strip()])


def count_chars(text):
    """Characters excluding whitespace and zero-width joiners."""
    t = text
    for ch in ZW:
        t = t.replace(ch, "")
    return len(re.sub(r"\s+", "", t))


def detect_script(text):
    """Return 'gurmukhi', 'shahmukhi', 'latin' or 'mixed' by codepoint majority."""
    g = a = l = 0
    for ch in text:
        cp = ord(ch)
        if GURMUKHI[0] <= cp <= GURMUKHI[1]:
            g += 1
        elif any(lo <= cp <= hi for lo, hi in ARABIC_BLOCKS):
            a += 1
        elif ("a" <= ch.lower() <= "z"):
            l += 1
    total = g + a + l
    if total == 0:
        return "unknown"
    counts = {"gurmukhi": g, "shahmukhi": a, "latin": l}
    top = max(counts, key=counts.get)
    return top if counts[top] / total > 0.7 else "mixed"


def _is_fragment(decoded):
    """True when a single token does not decode to a whole character."""
    return (decoded == "") or ("\ufffd" in decoded)


def fertility_stats(texts, tokenizer, unk_token_ids=None):
    """
    tokenizer: anything with .encode(text) -> list[int] and, optionally,
               .convert_ids_to_tokens(ids) -> list[str].

    Returns per-corpus aggregates plus per-sentence arrays for significance
    testing (the aggregates alone cannot support a claim of difference).
    """
    tps, tpw, tpc, unk_counts, byte_counts, frag_counts = [], [], [], [], [], []

    for t in texts:
        ids = tokenizer.encode(t)
        n_tok = len(ids)
        w = count_words(t)
        c = count_chars(t)

        tps.append(n_tok)
        tpw.append(n_tok / w if w else np.nan)
        tpc.append(n_tok / c if c else np.nan)

        if unk_token_ids:
            unk_counts.append(sum(1 for i in ids if i in unk_token_ids))
        else:
            unk_counts.append(0)

        # byte-fallback pieces look like <0xE0> in SentencePiece vocabularies
        try:
            toks = tokenizer.convert_ids_to_tokens(ids)
            byte_counts.append(sum(1 for tk in toks
                                   if isinstance(tk, str)
                                   and tk.startswith("<0x") and tk.endswith(">")))
        except Exception:
            byte_counts.append(0)

        # script-agnostic fragmentation: a token that decodes alone to an
        # incomplete character is a raw byte, whatever the vocabulary style
        try:
            frag_counts.append(sum(1 for i in ids
                                   if _is_fragment(tokenizer.decode([i]))))
        except Exception:
            frag_counts.append(0)

    tps = np.array(tps, float)
    tpw = np.array(tpw, float)
    tpc = np.array(tpc, float)
    unk = np.array(unk_counts, float)
    byt = np.array(byte_counts, float)
    frg = np.array(frag_counts, float)
    total_tok = tps.sum()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _summarise(texts, tps, tpw, tpc, unk, byt, frg, total_tok)


def _summarise(texts, tps, tpw, tpc, unk, byt, frg, total_tok):
    return {
        "n_sentences": len(texts),
        "tokens_per_sent_mean": float(np.nanmean(tps)),
        "tokens_per_word_mean": float(np.nanmean(tpw)),
        "tokens_per_char_mean": float(np.nanmean(tpc)),
        "tokens_per_sent_median": float(np.nanmedian(tps)),
        "tokens_per_word_median": float(np.nanmedian(tpw)),
        "tokens_per_char_median": float(np.nanmedian(tpc)),
        "unk_rate": float(unk.sum() / total_tok) if total_tok else np.nan,
        "byte_fallback_rate": float(byt.sum() / total_tok) if total_tok else np.nan,
        "fragment_rate": float(frg.sum() / total_tok) if total_tok else np.nan,
        "_per_sentence": {"tokens": tps, "tpw": tpw, "tpc": tpc},
    }


def compare_scripts(stats_a, stats_b, label_a="gurmukhi", label_b="shahmukhi"):
    """
    Paired comparison on ALIGNED sentences. Uses Wilcoxon signed-rank (paired,
    non-parametric) because fertility is right-skewed and the pairs are the
    same sentence in two scripts -- an unpaired t-test here would be wrong.
    """
    from scipy.stats import wilcoxon

    out = {"label_a": label_a, "label_b": label_b}
    for key in ("tokens", "tpw", "tpc"):
        a = stats_a["_per_sentence"][key]
        b = stats_b["_per_sentence"][key]
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]

        ratio = float(np.nanmean(b) / np.nanmean(a)) if np.nanmean(a) else np.nan
        if len(a) == 0:
            stat, p = np.nan, np.nan
        elif np.allclose(a, b):
            # identical corpora: no difference to test, p is exactly 1
            stat, p = 0.0, 1.0
        else:
            try:
                stat, p = wilcoxon(a, b)
                p = float(p)
            except Exception:
                stat, p = np.nan, np.nan

        out[key] = {
            f"mean_{label_a}": float(np.nanmean(a)),
            f"mean_{label_b}": float(np.nanmean(b)),
            "ratio_b_over_a": ratio,
            "wilcoxon_p": p,
            "n_pairs": int(len(a)),
        }
    return out
