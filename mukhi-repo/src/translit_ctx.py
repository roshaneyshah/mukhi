"""
Context-sensitive rule-based Shahmukhi -> Gurmukhi transliteration.

Replaces the naive character map as "system B" for audit disagreement
flagging. Developed on the PuMVR dev split only; the held-out test split was
scored once at the end (see results/FINDING_04).

What rules CAN fix: position-dependent vowel letters (alif, vao, ye), shadda
-> addak, aspirate digraphs with do-chashmi he, nasals before stops,
nun-ghunna tippi/bindi choice.

What no rule can fix: short vowels (sihari, aunkar) that Shahmukhi leaves
unwritten. ਟਿੱਕਾ is written ٹکا; the ਿ is simply not in the input. This sets a
floor on CER for any system without a lexicon or language model, and is the
reason the SLPG NMT model (which has one) is the stronger system A.
"""
import unicodedata

# ---- consonants -------------------------------------------------------------
CONS = {
    "ب": "ਬ", "پ": "ਪ", "ت": "ਤ", "ٹ": "ਟ", "ث": "ਸ",
    "ج": "ਜ", "چ": "ਚ", "ح": "ਹ", "خ": "ਖ਼", "د": "ਦ",
    "ڈ": "ਡ", "ذ": "ਜ਼", "ر": "ਰ", "ڑ": "ੜ", "ز": "ਜ਼",
    "ژ": "ਜ਼", "س": "ਸ", "ش": "ਸ਼", "ص": "ਸ", "ض": "ਜ਼",
    "ط": "ਤ", "ظ": "ਜ਼", "غ": "ਗ਼", "ف": "ਫ਼", "ق": "ਕ",
    "ک": "ਕ", "ك": "ਕ", "گ": "ਗ", "ل": "ਲ", "م": "ਮ",
    "ن": "ਨ", "ݨ": "ਣ", "ڻ": "ਣ", "ه": "ਹ", "ہ": "ਹ",
    "ھ": "ਹ",
}
# aspirates: consonant + do-chashmi he (U+06BE)
ASP = {"ਬ": "ਭ", "ਪ": "ਫ", "ਤ": "ਥ", "ਟ": "ਠ", "ਜ": "ਝ", "ਚ": "ਛ",
       "ਦ": "ਧ", "ਡ": "ਢ", "ਕ": "ਖ", "ਗ": "ਘ",
       "ਰ": "ਰ੍ਹ", "ੜ": "ੜ੍ਹ", "ਲ": "ਲ੍ਹ", "ਮ": "ਮ੍ਹ", "ਨ": "ਨ੍ਹ", "ਣ": "ਣ੍ਹ"}

ALIF, ALIF_MADDA = "ا", "آ"
VAO, HAMZA_VAO = "و", "ؤ"
YE, YE_ARABIC, BARI_YE, HAMZA_YE = "ی", "ي", "ے", "ئ"
NUN, NUN_GHUNNA = "ن", "ں"
AIN = "ع"
HE_GOAL, HE = "ہ", "ه"
DO_CHASHMI = "ھ"
SHADDA, ZABAR, ZER, PESH, SUKUN = "ّ", "َ", "ِ", "ُ", "ْ"
HAMZA = "ء"
DROP = set("ًٌٍٰٕٔ‌‍") | {SUKUN, ZABAR, HAMZA}

PUNCT = {"،": ",", "؟": "?", "۔": "।", "؛": ";"}
DIGITS = {chr(0x06F0 + i): chr(0x0A66 + i) for i in range(10)}
DIGITS.update({chr(0x0660 + i): chr(0x0A66 + i) for i in range(10)})

STOPS_FOR_TIPPI = set("کگچجٹڈتدپب")

LONG_MATRA = set("ਾੀੇੈੋੌ")
INDEP_LONG = set("ਆਈਏਐਓਔ")


def _is_cons(ch):
    return ch in CONS


def _nasal_mark(prev_out):
    """Bindi after long vowels, tippi otherwise (standard Gurmukhi convention)."""
    last = prev_out[-1] if prev_out else ""
    return "ਂ" if (last in LONG_MATRA or last in INDEP_LONG) else "ੰ"


DEFAULT_CONFIG = {
    "ye_medial": "ੀ",        # consonant + ye, not word-final: ੀ | ੇ | ੈ
    "vao_medial": "ੋ",       # consonant + vao: ੋ | ੂ | ੌ
    "nun_retroflex": "none",  # none | final_syllable | all_noninitial
    "he_final": "ਾ",         # word-final he after consonant: ਾ | ਹ | ੇ
    "mim_labial_tippi": True,   # mim before b/p -> tippi (ਲੰਬਾ)
}


def translit_word(w, cfg=None):
    cfg = dict(DEFAULT_CONFIG, **(cfg or {}))
    out = []            # list of Gurmukhi strings
    state = "start"     # start | cons | vowel
    i, n = 0, len(w)

    while i < n:
        ch = w[i]
        nxt = w[i + 1] if i + 1 < n else ""
        final = (i == n - 1)

        if ch in DROP:
            i += 1
            continue
        if ch in PUNCT:
            out.append(PUNCT[ch]); state = "start"; i += 1; continue
        if ch in DIGITS:
            out.append(DIGITS[ch]); state = "start"; i += 1; continue

        # short-vowel diacritics when written
        if ch == ZER:
            out.append("ਿ" if state == "cons" else "ਇ"); state = "vowel"; i += 1; continue
        if ch == PESH:
            out.append("ੁ" if state == "cons" else "ਉ"); state = "vowel"; i += 1; continue

        # ---- alif ---------------------------------------------------------
        if ch == ALIF_MADDA:
            out.append("ਆ" if state != "cons" else "ਾ"); state = "vowel"; i += 1; continue
        if ch == ALIF:
            if state == "cons":
                out.append("ਾ"); state = "vowel"; i += 1; continue
            # word-initial or after a vowel: look at what follows
            if nxt == ZER:
                out.append("ਇ"); state = "vowel"; i += 2; continue
            if nxt == PESH:
                out.append("ਉ"); state = "vowel"; i += 2; continue
            if nxt == VAO:
                out.append("ਉ" if state == "start" else "ਓ"); state = "vowel"; i += 2; continue
            if nxt in (YE, YE_ARABIC):
                out.append("ਇ" if state == "start" else "ਈ"); state = "vowel"; i += 2; continue
            if nxt == BARI_YE:
                out.append("ਐ" if state == "start" else "ਏ"); state = "vowel"; i += 2; continue
            out.append("ਅ" if state == "start" else "ਆ"); state = "vowel"; i += 1; continue

        # ---- vao ----------------------------------------------------------
        if ch in (VAO, HAMZA_VAO):
            if ch == HAMZA_VAO:
                out.append("ਉ"); state = "vowel"; i += 1; continue
            if state == "start" or nxt in (ALIF, ALIF_MADDA) or nxt == SHADDA:
                out.append("ਵ"); state = "cons"; i += 1; continue
            if state == "cons":
                out.append(cfg["vao_medial"]); state = "vowel"; i += 1; continue
            out.append("ਓ"); state = "vowel"; i += 1; continue

        # ---- hamza-ye carrier ---------------------------------------------
        if ch == HAMZA_YE:
            if nxt in (YE, YE_ARABIC):
                out.append("ਈ"); state = "vowel"; i += 2; continue
            if nxt == BARI_YE:
                out.append("ਏ"); state = "vowel"; i += 2; continue
            if nxt == VAO:
                out.append("ਓ"); state = "vowel"; i += 2; continue
            out.append("ਇ"); state = "vowel"; i += 1; continue

        # ---- ye -----------------------------------------------------------
        if ch in (YE, YE_ARABIC):
            if state == "start":
                out.append("ਯ"); state = "cons"; i += 1; continue
            if state == "cons":
                if nxt in (ALIF, ALIF_MADDA):          # consonant+ye+alif -> ਿਆ
                    out.append("ਿਆ"); state = "vowel"; i += 2; continue
                out.append("ੀ" if final else cfg["ye_medial"]); state = "vowel"; i += 1; continue
            out.append("ਈ" if final else "ਯ")
            state = "vowel" if final else "cons"; i += 1; continue

        if ch == BARI_YE:
            out.append("ੇ" if state == "cons" else "ਏ"); state = "vowel"; i += 1; continue

        # ---- nasals -------------------------------------------------------
        if ch == NUN_GHUNNA:
            out.append(_nasal_mark("".join(out))); i += 1; continue
        if ch == NUN and state != "start" and nxt in STOPS_FOR_TIPPI:
            out.append(_nasal_mark("".join(out))); i += 1; continue
        if (ch == "\u0645" and cfg["mim_labial_tippi"] and state != "start"
                and nxt in ("\u0628", "\u067e")):
            out.append(_nasal_mark("".join(out))); i += 1; continue
        if ch == NUN and state == "vowel":
            mode = cfg["nun_retroflex"]
            rest = w[i + 1:]
            if mode == "all_noninitial" or (
                    mode == "final_syllable"
                    and (rest == "" or rest in (ALIF, YE, BARI_YE, YE_ARABIC))):
                out.append("ਣ"); state = "cons"; i += 1; continue

        # ---- ain: treated as a vowel carrier ------------------------------
        if ch == AIN:
            if state == "start":
                out.append("ਅ"); state = "vowel"
            i += 1; continue

        # ---- final he after a consonant is usually a vowel ending ---------
        if ch in (HE_GOAL, HE) and final and state == "cons":
            out.append(cfg["he_final"]); state = "vowel" if cfg["he_final"] != "ਹ" else "cons"
            i += 1; continue

        # ---- consonants (with aspiration and gemination) ------------------
        if _is_cons(ch):
            g = CONS[ch]
            j = i + 1
            gem = False
            while j < n and w[j] in (SHADDA, ZABAR, SUKUN):
                gem = gem or w[j] == SHADDA
                j += 1
            if j < n and w[j] == DO_CHASHMI and g in ASP:
                g = ASP[g]; j += 1
                while j < n and w[j] == SHADDA:
                    gem = True; j += 1
            if gem and state != "start":
                out.append("ੱ")
            out.append(g)
            state = "cons"
            i = j
            continue

        # anything else (Latin, hyphen, Gurmukhi already) passes through
        out.append(ch)
        state = "start" if not ch.isalnum() else state
        i += 1

    return unicodedata.normalize("NFC", "".join(out))


# Selected by grid search on the PuMVR DEV split (see tune_config and
# results/FINDING_04). Overwritten below once tuning has been run.
TUNED_CONFIG = dict(DEFAULT_CONFIG)


def ctx_shah_to_gur(text, cfg=None):
    """Transliterate running text word by word, preserving spacing and hyphens."""
    if not text:
        return ""
    cfg = TUNED_CONFIG if cfg is None else cfg
    words = []
    for tok in text.split(" "):
        parts = tok.split("-")
        words.append("-".join(translit_word(p, cfg) for p in parts))
    return " ".join(words)


def tune_config(dev_pairs, cer_fn):
    """Exhaustive search over the ambiguous defaults, scored on DEV only."""
    import itertools
    grid = {
        "ye_medial": ["ੀ", "ੇ", "ੈ"],
        "vao_medial": ["ੋ", "ੂ", "ੌ"],
        "nun_retroflex": ["none", "final_syllable", "all_noninitial"],
        "he_final": ["ਾ", "ਹ", "ੇ"],
        "mim_labial_tippi": [True, False],
    }
    keys = list(grid)
    refs = [g for g, s in dev_pairs]
    results = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        cfg = dict(zip(keys, combo))
        hyps = [ctx_shah_to_gur(s, cfg) for g, s in dev_pairs]
        results.append((cer_fn(hyps, refs)["cer"], cfg))
    results.sort(key=lambda t: t[0])
    return results
