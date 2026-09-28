"""
Experimental conditions: every text set is aligned item-by-item to the same
underlying sentences, so any pair of conditions forms a contrast.

The contrast table (see CONTRASTS) extends the proposal's four rows with an
information-matched control that FINDING_04 made possible: stripping the
characters Shahmukhi leaves unwritten (sihari, aunkar, addak) from Gurmukhi
gives a Gurmukhi text carrying roughly Shahmukhi's information. That splits
the Punjabi script gap into an INFORMATION part and a SCRIPT-SHAPE part.
"""
import unicodedata

# ---------------------------------------------------------------------------
# Serbian Cyrillic -> Latin: the standard, lossless, one-to-one mapping.
# Three Cyrillic letters map to Latin digraphs (lj, nj, dz-caron).
# ---------------------------------------------------------------------------
_SR = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ђ": "đ", "е": "e",
    "ж": "ž", "з": "z", "и": "i", "ј": "j", "к": "k", "л": "l", "љ": "lj",
    "м": "m", "н": "n", "њ": "nj", "о": "o", "п": "p", "р": "r", "с": "s",
    "т": "t", "ћ": "ć", "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "č",
    "џ": "dž", "ш": "š",
}


def serbian_cyr_to_lat(text):
    """
    Lossless Serbian Cyrillic -> Latin. Case rule for digraphs: an uppercase
    Љ/Њ/Џ becomes 'LJ' if the next letter is also uppercase (all-caps word),
    otherwise 'Lj'.
    """
    out = []
    n = len(text)
    for i, ch in enumerate(text):
        low = ch.lower()
        if low not in _SR:
            out.append(ch)
            continue
        lat = _SR[low]
        if ch != low:                       # uppercase source letter
            nxt = text[i + 1] if i + 1 < n else ""
            if len(lat) == 2 and nxt.isalpha() and nxt.isupper():
                lat = lat.upper()
            else:
                lat = lat[0].upper() + lat[1:]
        out.append(lat)
    return "".join(out)


# ---------------------------------------------------------------------------
# Information-matched Gurmukhi
# ---------------------------------------------------------------------------
UNWRITTEN_IN_SHAHMUKHI = {"ਿ", "ੁ", "ੱ"}   # sihari, aunkar, addak


def devowel_gurmukhi(text):
    """Remove the characters running Shahmukhi does not write. Result is still
    Gurmukhi script but carries (approximately) Shahmukhi's information."""
    t = unicodedata.normalize("NFC", text or "")
    return "".join(ch for ch in t if ch not in UNWRITTEN_IN_SHAHMUKHI)


# ---------------------------------------------------------------------------
# Romanisation
# ---------------------------------------------------------------------------
_UROMAN = None


def romanize(texts, lcode):
    """uroman romanisation (pip install uroman). lcode: ISO 639-3."""
    global _UROMAN
    if _UROMAN is None:
        import uroman as ur
        _UROMAN = ur.Uroman()
    return [_UROMAN.romanize_string(t, lcode=lcode) for t in texts]


# ---------------------------------------------------------------------------
# FLORES+  (gated on Hugging Face: accept terms, set HF_TOKEN)
# ---------------------------------------------------------------------------
FLORES_CODES = ("pan_Guru", "urd_Arab", "hin_Deva", "srp_Cyrl")


def load_flores(codes=FLORES_CODES, splits=("dev", "devtest"), cache_dir=None):
    """
    Returns {code: [texts]} aligned across codes by FLORES id.
    Alignment is verified by id, never assumed from row order.
    """
    from datasets import load_dataset
    by_code = {}
    for code in codes:
        rows = []
        for sp in splits:
            ds = load_dataset("openlanguagedata/flores_plus", code, split=sp,
                              cache_dir=cache_dir)
            rows.extend((f"{sp}:{r['id']}", r["text"]) for r in ds)
        by_code[code] = dict(rows)

    ids = sorted(set.intersection(*(set(v) for v in by_code.values())))
    if not ids:
        raise RuntimeError("no FLORES ids shared across all requested codes")
    return ids, {c: [by_code[c][i] for i in ids] for c in codes}


def build_conditions(ids, flores, gur_to_shah=None, rom=True):
    """
    flores: {code: texts} from load_flores.
    gur_to_shah: Gurmukhi -> Shahmukhi callable. Defaults to the rule system
        in translit_g2s (5.6% held-out CER, FINDING_06). FLORES has no
        Shahmukhi, so this condition is machine-generated: say so in the
        paper, and use the human-written PuMVR set as the robustness check.

    Returns {condition_name: [texts]} all aligned to `ids`.
    """
    if gur_to_shah is None:
        from translit_g2s import gur_to_shah
    g = flores["pan_Guru"]
    c = {
        "pan_Guru": g,
        "pan_Guru_devowel": [devowel_gurmukhi(t) for t in g],
        "pan_Shah": [gur_to_shah(t) for t in g],
        "urd_Arab": flores["urd_Arab"],
        "hin_Deva": flores["hin_Deva"],
    }
    if "srp_Cyrl" in flores:
        c["srp_Cyrl"] = flores["srp_Cyrl"]
        c["srp_Latn"] = [serbian_cyr_to_lat(t) for t in flores["srp_Cyrl"]]
    if rom:
        c["pan_Guru_rom"] = romanize(c["pan_Guru"], "pan")
        c["pan_Shah_rom"] = romanize(c["pan_Shah"], "pnb")
        c["urd_rom"] = romanize(c["urd_Arab"], "urd")
    for k, v in c.items():
        if len(v) != len(ids):
            raise RuntimeError(f"condition {k} has {len(v)} items, expected {len(ids)}")
    return c


# (a, b, what varies, what is held constant)
CONTRASTS = [
    ("pan_Guru", "pan_Shah", "script + information", "language, content"),
    ("pan_Guru", "pan_Guru_devowel", "information only", "language, content, script"),
    ("pan_Guru_devowel", "pan_Shah", "script shape (information ~matched)", "language, content"),
    ("pan_Shah", "urd_Arab", "language", "script, content"),
    ("hin_Deva", "urd_Arab", "script (+ register confound)", "content"),
    ("pan_Guru_rom", "pan_Shah_rom", "information only, in shared Latin script", "language, content"),
    ("pan_Shah_rom", "urd_rom", "language, script removed", "content"),
    ("srp_Cyrl", "srp_Latn", "script (balanced, lossless)", "language, content"),
]
