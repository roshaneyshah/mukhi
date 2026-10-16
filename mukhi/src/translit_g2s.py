"""
Rule-based Gurmukhi -> Shahmukhi.

This is the easy direction: Gurmukhi writes every vowel Shahmukhi needs, so
the job is mostly deletion (short vowels) and mapping. Remaining ambiguity is
in Arabic-origin spellings (ਸ as س/ص/ث, ਜ਼ as ز/ذ/ض/ظ, ਤ as ت/ط), which the rules
resolve to the commonest letter, and in whether writers mark gemination.

Used to build the Shahmukhi condition from FLORES pan_Guru, since FLORES has
no Shahmukhi and the SLPG NMT models are fairseq checkpoints with undocumented
preprocessing. Measured on PuMVR in results/FINDING_06: 5.6% CER on held-out
words (83% exact), 6.4% on 1,000 sentences. Defaults below were selected on
the dev split only.
"""
import unicodedata

C = {
    "ਕ": "ک", "ਖ": "کھ", "ਗ": "گ", "ਘ": "گھ", "ਙ": "نگ", "ਚ": "چ", "ਛ": "چھ",
    "ਜ": "ج", "ਝ": "جھ", "ਞ": "نج", "ਟ": "ٹ", "ਠ": "ٹھ", "ਡ": "ڈ", "ਢ": "ڈھ",
    "ਣ": "ݨ", "ਤ": "ت", "ਥ": "تھ", "ਦ": "د", "ਧ": "دھ", "ਨ": "ن", "ਪ": "پ",
    "ਫ": "پھ", "ਬ": "ب", "ਭ": "بھ", "ਮ": "م", "ਯ": "ی", "ਰ": "ر", "ਲ": "ل",
    "ਵ": "و", "ੜ": "ڑ", "ਸ": "س", "ਹ": "ہ",
    "ਸ਼": "ش", "ਖ਼": "خ", "ਗ਼": "غ", "ਜ਼": "ز", "ਫ਼": "ف", "ਲ਼": "ل",
}
NUKTA = "਼"
NUKTA_FORMS = {"ਸ": "ਸ਼", "ਖ": "ਖ਼", "ਗ": "ਗ਼", "ਜ": "ਜ਼", "ਫ": "ਫ਼", "ਲ": "ਲ਼"}

INDEP_INITIAL = {"ਅ": "ا", "ਆ": "آ", "ਇ": "ا", "ਈ": "ای", "ਉ": "ا", "ਊ": "او",
                 "ਏ": "اے", "ਐ": "اے", "ਓ": "او", "ਔ": "او"}
INDEP_MEDIAL = {"ਅ": "ا", "ਆ": "ا", "ਇ": "ئ", "ਈ": "ئی", "ਉ": "ؤ", "ਊ": "ؤ",
                "ਏ": "ئے", "ਐ": "ئے", "ਓ": "و", "ਔ": "و"}
MATRA = {"ਾ": "ا", "ਿ": "", "ੀ": "ی", "ੁ": "", "ੂ": "و", "ੋ": "و", "ੌ": "و"}
MATRA_E = {"ੇ", "ੈ"}
TIPPI, BINDI, ADDAK, VIRAMA = "ੰ", "ਂ", "ੱ", "੍"
PUNCT = {"।": "۔", "?": "؟", ",": "،", ";": "؛"}
DIGITS = {chr(0x0A66 + i): chr(0x06F0 + i) for i in range(10)}

DEFAULT_G2S = {"nn": "ن", "mark_gemination": False, "e_medial": "ی", "digits": "extended"}


def _compose(text):
    """NFC decomposes nukta letters; recombine them so lookups see ਸ਼ etc."""
    t = unicodedata.normalize("NFC", text)
    out, i = [], 0
    while i < len(t):
        if i + 1 < len(t) and t[i + 1] == NUKTA and t[i] in NUKTA_FORMS:
            out.append(NUKTA_FORMS[t[i]]); i += 2
        else:
            out.append(t[i]); i += 1
    return out


def g2s_word(w, cfg=None):
    cfg = dict(DEFAULT_G2S, **(cfg or {}))
    ch = _compose(w)
    n = len(ch)
    out, prev = [], "start"     # start | cons | vowel
    i = 0
    while i < n:
        c = ch[i]
        final = i == n - 1
        if c in C:
            g = C[c]
            if c == "ਣ":
                g = cfg["nn"]
            if i + 2 < n and ch[i + 1] == VIRAMA and ch[i + 2] in ("ਹ", "ਰ", "ਵ"):
                g += {"ਹ": "ھ", "ਰ": "ر", "ਵ": "و"}[ch[i + 2]]
                i += 2
            if prev == "cons" and out and out[-1] == ADDAK:
                out.pop()
                out.append(g + ("ّ" if cfg["mark_gemination"] else ""))
            else:
                out.append(g)
            prev = "cons"
        elif c == ADDAK:
            out.append(ADDAK); prev = "cons"         # placeholder, resolved at next consonant
        elif c == "ਿ" and i + 1 < n and ch[i + 1] in ("ਆ", "ਅ", "ਓ", "ਉ", "ਏ"):
            out.append("ی"); prev = "vowel"          # ਹੋਸ਼ਿਆਰ -> ہوشیار
        elif c in MATRA:
            out.append(MATRA[c]); prev = "vowel" if MATRA[c] else prev
        elif c in MATRA_E:
            out.append("ے" if final else cfg["e_medial"]); prev = "vowel"
        elif c in INDEP_INITIAL:
            if prev == "start":
                g = INDEP_INITIAL[c]
                if c in ("ਏ", "ਐ") and not final:
                    g = "ای"                                   # ਐਂਬੂਲੈਂਸ -> ایمبولینس
            else:
                g = INDEP_MEDIAL[c]
                if c == "ਇ" and i + 1 < n and ch[i + 1] in C:
                    g = "ی"                                    # ਆਇਤ -> آیت
            out.append(g); prev = "vowel"
        elif c in (TIPPI, BINDI):
            nxt = ch[i + 1] if i + 1 < n else ""
            if nxt in ("ਨ", "ਮ"):
                pass                                            # ਕੰਨ -> کن: gemination, not a nasal
            elif nxt in ("ਬ", "ਪ", "ਭ", "ਫ"):
                out.append("م")                                # ਲੰਬਾ -> لمبا
            else:
                out.append("ں" if final else "ن")
        elif c in PUNCT:
            out.append(PUNCT[c]); prev = "start"
        elif c in DIGITS:
            out.append(DIGITS[c] if cfg["digits"] == "extended" else str(ord(c) - 0x0A66))
            prev = "start"
        elif c == VIRAMA:
            pass
        else:
            out.append(c); prev = "start" if not c.isalnum() else prev
        i += 1
    return "".join(x for x in out if x != ADDAK)


def gur_to_shah(text, cfg=None):
    if not text:
        return ""
    return " ".join("-".join(g2s_word(p, cfg) for p in tok.split("-"))
                    for tok in text.split(" "))


# ---- evaluation in Perso-Arabic script ---------------------------------------
_AR_UNIFY = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک",
                           "ه": "ہ", "ۃ": "ہ", "ة": "ہ"})
_AR_DIACRITICS = set("ًٌٍَُِّْٰٕٔ")


def normalise_shah(t, strip_diacritics=True):
    """Unify Arabic/Urdu code-point variants (ی/ي, ک/ك, ہ/ه) on both sides;
    optionally drop harakat, which writers use inconsistently."""
    t = unicodedata.normalize("NFC", " ".join((t or "").split())).translate(_AR_UNIFY)
    if strip_diacritics:
        t = "".join(c for c in t if c not in _AR_DIACRITICS)
    return t
