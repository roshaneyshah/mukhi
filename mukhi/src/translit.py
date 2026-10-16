"""
Transliteration systems for the audit's disagreement flagging.

System A: the SLPG NMT model (`load_nmt_transliterator`), 87.7 BLEU in the
Shahmukhi -> Gurmukhi direction.

System B: `translit_ctx.ctx_shah_to_gur`, context-sensitive rules measured at
16.0% character error rate on held-out PuMVR words (results/FINDING_04).

`rule_based_shah_to_gur` below is the ORIGINAL naive character map, kept only
as the documented baseline (50.9% held-out CER). Do not use it for anything
else.
"""

# Consonants and long vowels. Short vowels (zabar/zer/pesh) are usually absent
# in running Shahmukhi text, which is the principal source of ambiguity.
SHAH_TO_GUR = {
    "ا": "ਅ",  # alif      -> a
    "آ": "ਆ",  # alif madda-> aa
    "ب": "ਬ",  # be        -> ba
    "پ": "ਪ",  # pe        -> pa
    "ت": "ਤ",  # te        -> ta
    "ٹ": "ਟ",  # tte       -> tta
    "ث": "ਸ",  # se        -> sa
    "ج": "ਜ",  # jim       -> ja
    "چ": "ਚ",  # che       -> cha
    "ح": "ਹ",  # he        -> ha
    "خ": "ਖ",  # khe       -> kha
    "د": "ਦ",  # dal       -> da
    "ڈ": "ਡ",  # ddal      -> dda
    "ذ": "ਜ",  # zal       -> za
    "ر": "ਰ",  # re        -> ra
    "ڑ": "ੜ",  # rre       -> rra
    "ز": "ਜ",  # ze        -> za
    "ژ": "ਜ",  # zhe       -> zha
    "س": "ਸ",  # sin       -> sa
    "ش": "ਸ਼",  # shin      -> sha
    "ص": "ਸ",  # sad       -> sa
    "ض": "ਜ",  # zad       -> za
    "ط": "ਤ",  # toe       -> ta
    "ظ": "ਜ",  # zoe       -> za
    "ع": "ਅ",  # ain       -> a
    "غ": "ਗ",  # ghain     -> gha
    "ف": "ਫ",  # fe        -> pha
    "ق": "ਕ",  # qaf       -> ka
    "ک": "ਕ",  # kaf       -> ka
    "گ": "ਗ",  # gaf       -> ga
    "ل": "ਲ",  # lam       -> la
    "م": "ਮ",  # mim       -> ma
    "ن": "ਨ",  # nun       -> na
    "ں": "ੰ",  # nun ghunna-> tippi
    "و": "ਵ",  # vao       -> va
    "ہ": "ਹ",  # he goal   -> ha
    "ۃ": "ਹ",
    "ی": "ਯ",  # ye        -> ya
    "ے": "ੇ",  # bari ye   -> e
    "ء": "",        # hamza     -> (dropped)
    "ٔ": "",
    "ً": "", "ٌ": "", "ٍ": "",   # tanwin
    "َ": "", "ُ": "", "ِ": "",   # short vowels
    "ّ": "", "ْ": "",                  # shadda, sukun
    "‌": "", "‍": "",                  # zero-width
}


def rule_based_shah_to_gur(text):
    """Naive character-by-character map. Scaffolding only -- see module docstring."""
    if not text:
        return ""
    return "".join(SHAH_TO_GUR.get(ch, ch) for ch in text)


def load_slpg_fairseq(direction="g2s", cache_dir=None, beam=5):
    """
    SLPG transliteration models are FAIRSEQ checkpoints, not transformers
    models (an earlier version of this repo wrongly used AutoModelForSeq2SeqLM).
    Their input preprocessing is undocumented, so this loader relies on
    whatever tokenizer/BPE the checkpoint's own cfg declares, and the result
    MUST pass `score_on_pumvr` before being used for anything. Never trust it
    blind.

    Requires `pip install fairseq` (often fails on Python >= 3.11; try
    `pip install fairseq==0.12.2 --no-build-isolation` or a 3.10 kernel).
    Returns a callable str -> str.
    """
    from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
    from fairseq.sequence_generator import SequenceGenerator

    repo = {"g2s": "SLPG/Punjabi_Gurmukhi_to_Shahmukhi_Transliteration",
            "s2g": "SLPG/Punjabi_Shahmukhi_to_Gurmukhi_Transliteration"}[direction]
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
        repo, cache_dir=cache_dir)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = [m.to(dev).eval() for m in models]
    gen = SequenceGenerator(models, task.target_dictionary, beam_size=beam)
    src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

    avg_len = sum(len(src_dict[i]) for i in range(src_dict.nspecial, len(src_dict))) / \
        max(len(src_dict) - src_dict.nspecial, 1)
    char_level = avg_len < 1.5

    def run(text):
        if not text or not text.strip():
            return ""
        inp = " ".join(text.replace(" ", "\u2581")) if char_level else text
        toks = src_dict.encode_line(inp, add_if_not_exist=False, append_eos=True).long()
        sample = {"net_input": {"src_tokens": toks.unsqueeze(0).to(dev),
                                "src_lengths": torch.tensor([toks.numel()]).to(dev)}}
        with torch.no_grad():
            hyp = gen.generate(models, sample)[0][0]["tokens"]
        out = tgt_dict.string(hyp)
        return out.replace(" ", "").replace("\u2581", " ") if char_level else out

    run.char_level = char_level
    return run


def score_on_pumvr(fn, pumvr_json, direction="g2s", n=300):
    """Score a transliterator against human PuMVR sentences. Returns CER.
    Used as a gate: the NMT replaces the rules only if it is measurably better."""
    from pumvr import load_pumvr_pairs
    from translit_eval import levenshtein, normalise_gur
    from translit_g2s import normalise_shah
    S, _, _ = load_pumvr_pairs(pumvr_json)
    S = S[:n]
    if direction == "g2s":
        hyp = [normalise_shah(fn(x["gurmukhi"])) for x in S]
        ref = [normalise_shah(x["shahmukhi"]) for x in S]
    else:
        hyp = [normalise_gur(fn(x["shahmukhi"])) for x in S]
        ref = [normalise_gur(x["gurmukhi"]) for x in S]
    e = sum(levenshtein(h, r) for h, r in zip(hyp, ref))
    return e / max(sum(len(r) for r in ref), 1)


def load_nmt_transliterator(cache_dir=None, **kw):
    """Back-compat name: the Shahmukhi -> Gurmukhi fairseq model."""
    return load_slpg_fairseq("s2g", cache_dir=cache_dir, **kw)
