#!/usr/bin/env python3
"""Measure script resource directly, where the training data is public.

The paper's resource account was a hypothesis: Shahmukhi is likely rarer than
Gurmukhi in pretraining data. This tool measures what can be measured.

1. CC-100, the corpus released as XLM-R's training data. Each language file
   is streamed and every line is assigned a script by majority of letters.
   Arabic-script lines are further split into Punjabi (Shahmukhi) and Urdu by
   function words, a classifier validated on FLORES and reported with its
   accuracy. Serbian lines are split into Cyrillic and Latin.
2. Wikipedia article counts for the Gurmukhi (pa) and Shahmukhi (pnb)
   Punjabi editions, Urdu, and Serbian (sr, sh). MuRIL trained partly on
   Wikipedia.
3. Tokeniser vocabularies of every model: how many entries contain each
   script. A proxy for exposure, available even where data is not public.

Every section is independent and records its own failure. Output:
results/resource_counts.json. CPU and network only.
"""
import json
import lzma
import os
import re
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("MUKHI_RESULTS", os.path.join(ROOT, "results"))
OUT = os.path.join(RESULTS, os.environ.get("MUKHI_RESOURCE_OUT", "resource_counts.json"))

CC100_URL = "https://data.statmt.org/cc-100/{}.txt.xz"
CC100_LANGS = os.environ.get("MUKHI_CC100_LANGS", "pa,pnb,ur,sr").split(",")
# Sections to run: validation, wikipedia, vocab, cc100.
SECTIONS = os.environ.get("MUKHI_SECTIONS", "validation,wikipedia,vocab,cc100").split(",")
TIME_CAP = float(os.environ.get("MUKHI_CC100_SECONDS", "2400"))

SCRIPTS = {
    "gurmukhi": [(0x0A00, 0x0A7F)],
    "arabic": [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "cyrillic": [(0x0400, 0x04FF)],
    "latin": [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)],
    "devanagari": [(0x0900, 0x097F)],
}


def script_of(ch):
    o = ord(ch)
    for name, ranges in SCRIPTS.items():
        for a, b in ranges:
            if a <= o <= b:
                return name
    return None


def line_script(line):
    counts = {}
    for ch in line:
        s = script_of(ch)
        if s:
            counts[s] = counts.get(s, 0) + 1
    if not counts:
        return None, 0
    s = max(counts, key=counts.get)
    return s, sum(counts.values())


# Function words. Punjabi and Urdu share the script but not these.
PUNJABI_WORDS = {"دا", "دی", "دے", "نوں", "وچ", "توں", "اے", "ہن", "ایہہ", "اوہ",
                 "کیتا", "ہیگا", "ساڈا", "تہاڈا", "جیہڑا", "سن", "گیا", "وی", "تے",
                 "ہور", "اوہناں", "ایس"}
URDU_WORDS = {"کا", "کی", "کے", "کو", "میں", "ہے", "ہیں", "سے", "یہ", "وہ", "اور",
              "تھا", "تھی", "تھے", "بھی", "گیا", "کیا", "پر", "لیے", "ایک"}
_TOKEN = re.compile(r"[؀-ۿݐ-ݿ]+")


# Ekavian and ijekavian reflexes of the same words. Standard Serbian in Serbia
# is ekavian. Croatian, Bosnian and Montenegrin are ijekavian, and so is the
# Serbian written in Bosnia, so an ekavian count is a LOWER bound on Serbian.
EKAVIAN = {"deo", "dela", "vreme", "mesto", "mesta", "reč", "reči", "lepo",
           "svet", "sveta", "dete", "deca", "gde", "pre", "posle", "uvek",
           "mleko", "pesma", "pesme", "vek", "veka", "reka", "reke", "cena",
           "cene", "verovatno", "videti", "razumeti", "smer", "beli", "telo",
           "mera", "primer", "vesti", "vest", "ceo", "celo", "delo", "mesec",
           "meseca", "svetski", "ovde", "rešenje", "nedelja", "nedelje",
           "zvezda", "sedište", "primera", "mere", "svetu", "mestu", "vremenu"}
IJEKAVIAN = {"dio", "dijela", "vrijeme", "mjesto", "mjesta", "riječ", "riječi",
             "lijepo", "svijet", "svijeta", "dijete", "djeca", "gdje", "prije",
             "poslije", "uvijek", "mlijeko", "pjesma", "pjesme", "vijek", "vijeka",
             "rijeka", "rijeke", "cijena", "cijene", "vjerojatno", "vjerovatno",
             "vidjeti", "razumjeti", "smjer", "bijeli", "tijelo",
             "mjera", "primjer", "vijesti", "vijest", "cijeli", "cijelo", "djelo",
             "mjesec", "mjeseca", "svjetski", "ovdje", "rješenje", "nedjelja",
             "nedjelje", "zvijezda", "sjedište", "primjera", "mjere", "svijetu",
             "mjestu"}
_LTOKEN = re.compile(r"[a-zčćšžđ]+")


def latin_variety(line):
    """'ekavian', 'ijekavian' or None, by reflex votes."""
    toks = _LTOKEN.findall(line.lower())
    e = sum(t in EKAVIAN for t in toks)
    j = sum(t in IJEKAVIAN for t in toks)
    if e == j:
        return None
    return "ekavian" if e > j else "ijekavian"


def arabic_variety(line):
    """'punjabi', 'urdu' or None, by function-word votes."""
    toks = _TOKEN.findall(line)
    p = sum(t in PUNJABI_WORDS and t not in URDU_WORDS for t in toks)
    u = sum(t in URDU_WORDS and t not in PUNJABI_WORDS for t in toks)
    # ݨ (U+0768) is written in Shahmukhi and not in Urdu.
    p += 2 * line.count("ݨ")
    if p == u:
        return None
    return "punjabi" if p > u else "urdu"


def validate_classifier():
    """Accuracy on FLORES: generated Shahmukhi against Urdu, same sentences."""
    path = os.path.join(RESULTS, "conditions_flores.json")
    if not os.path.exists(path):
        return {"status": "conditions_flores.json absent"}
    c = json.load(open(path))["conditions"]
    out = {}
    for cond, want in (("pan_Shah", "punjabi"), ("urd_Arab", "urdu")):
        got = [arabic_variety(t) for t in c[cond]]
        n = len(got)
        out[cond] = {"n": n,
                     "correct": sum(g == want for g in got) / n,
                     "wrong": sum(g not in (None, want) for g in got) / n,
                     "undecided": sum(g is None for g in got) / n}
    # Ekavian against ijekavian, on FLORES Serbian (Latin, from Cyrillic)
    # against Croatian and Bosnian.
    try:
        c2 = json.load(open(path))["conditions"]
        sys.path.insert(0, os.path.join(ROOT, "src"))
        from conditions import load_flores
        _, fl = load_flores(codes=("hrv_Latn", "bos_Latn"))
        for name, texts, want in (("srp_Latn", c2["srp_Latn"], "ekavian"),
                                  ("hrv_Latn", fl["hrv_Latn"], "ijekavian"),
                                  ("bos_Latn", fl["bos_Latn"], "ijekavian")):
            got = [latin_variety(t) for t in texts]
            n = len(got)
            out[name] = {"n": n, "correct": got.count(want) / n,
                         "wrong": sum(g not in (None, want) for g in got) / n,
                         "undecided": got.count(None) / n}
    except Exception as e:
        out["latin_validation"] = {"status": f"unavailable: {str(e)[:160]}"}
    # Human-written Shahmukhi: short questions, so more lines go undecided.
    try:
        sys.path.insert(0, os.path.join(ROOT, "src"))
        from pumvr import fetch_pumvr, load_pumvr_pairs
        S, _, _ = load_pumvr_pairs(fetch_pumvr(os.path.join(ROOT, "data", "pumvr")))
        got = [arabic_variety(x["shahmukhi"]) for x in S]
        n = len(got)
        out["pumvr_human_shahmukhi"] = {
            "n": n, "correct": got.count("punjabi") / n,
            "wrong": got.count("urdu") / n, "undecided": got.count(None) / n}
    except Exception as e:
        out["pumvr_human_shahmukhi"] = {"status": f"unavailable: {str(e)[:120]}"}
    return out


def count_stream(fh, deadline):
    """Count lines and letters by script, and Arabic lines by variety."""
    lines = {}
    letters = {}
    arab = {"punjabi": 0, "urdu": 0, "undecided": 0}
    arab_letters = {"punjabi": 0, "urdu": 0, "undecided": 0}
    lat = {"ekavian": 0, "ijekavian": 0, "undecided": 0}
    lat_letters = {"ekavian": 0, "ijekavian": 0, "undecided": 0}
    n_bytes = 0
    complete = True
    for raw in fh:
        n_bytes += len(raw)
        line = raw.decode("utf-8", "ignore")
        s, n = line_script(line)
        if s:
            lines[s] = lines.get(s, 0) + 1
            letters[s] = letters.get(s, 0) + n
            if s == "arabic":
                v = arabic_variety(line) or "undecided"
                arab[v] += 1
                arab_letters[v] += n
            elif s == "latin":
                v = latin_variety(line) or "undecided"
                lat[v] += 1
                lat_letters[v] += n
        if n_bytes % (1 << 24) < len(raw) and time.time() > deadline:
            complete = False
            break
    return {"bytes_read": n_bytes, "complete": complete, "lines": lines,
            "letters": letters, "arabic_lines_by_variety": arab,
            "arabic_letters_by_variety": arab_letters,
            "latin_lines_by_variety": lat, "latin_letters_by_variety": lat_letters}


class _Counted:
    """File wrapper that counts compressed bytes, so a stream cut off by the
    time cap still reports what fraction of the file it covered."""
    def __init__(self, f):
        self.f, self.n = f, 0

    def read(self, *a):
        b = self.f.read(*a)
        self.n += len(b)
        return b


def _one_lang(lang):
    url = CC100_URL.format(lang)
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(url, timeout=60)
    except Exception as e:
        return lang, {"status": f"not available: {type(e).__name__}: {str(e)[:120]}"}
    total = resp.headers.get("Content-Length")
    raw = _Counted(resp)
    try:
        with lzma.open(raw) as fh:
            r = count_stream(fh, t0 + TIME_CAP)
    except Exception as e:
        return lang, {"status": f"failed: {type(e).__name__}: {str(e)[:160]}"}
    r["seconds"] = round(time.time() - t0)
    r["compressed_bytes_read"] = raw.n
    r["compressed_bytes_total"] = int(total) if total else None
    r["fraction_of_file"] = (raw.n / int(total)) if total else None
    r["status"] = "ok"
    return lang, r


def cc100():
    """Every language file at once, one process each."""
    from concurrent.futures import ProcessPoolExecutor
    out = {}
    with ProcessPoolExecutor(max_workers=max(1, len(CC100_LANGS))) as ex:
        for lang, r in ex.map(_one_lang, CC100_LANGS):
            out[lang] = r
            if r.get("status") == "ok":
                print(f"  cc100 {lang}: {r['bytes_read'] / 1e9:.2f} GB read, "
                      f"fraction {r['fraction_of_file']}, lines={r['lines']}", flush=True)
            else:
                print(f"  cc100 {lang}: {r['status']}", flush=True)
    return out


def wikipedia():
    out = {}
    for wiki in ("pa", "pnb", "ur", "sr", "sh"):
        url = (f"https://{wiki}.wikipedia.org/w/api.php?action=query&meta=siteinfo"
               f"&siprop=statistics&format=json")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mukhi-research/1.0"})
            st = json.load(urllib.request.urlopen(req, timeout=60))["query"]["statistics"]
            out[wiki] = {k: st.get(k) for k in ("articles", "pages", "edits")}
            print(f"  wikipedia {wiki}: {out[wiki]}", flush=True)
        except Exception as e:
            out[wiki] = {"status": f"failed: {type(e).__name__}: {str(e)[:120]}"}
    return out


VOCAB_MODELS = ["xlm-roberta-base", "google/muril-base-cased", "Qwen/Qwen2.5-1.5B",
                "Qwen/Qwen2.5-3B", "google/gemma-2-2b"]


def vocab_scripts(tok):
    """Vocabulary entries containing each script, after decoding each entry."""
    counts = {k: 0 for k in SCRIPTS}
    size = len(tok)
    for i in range(size):
        try:
            s = tok.decode([i])
        except Exception:
            continue
        seen = {script_of(ch) for ch in s} - {None}
        for k in seen:
            counts[k] += 1
    return {"vocab_size": size, "entries_with_script": counts}


def vocabularies(cache_dir=None):
    out = {}
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        return {"status": f"transformers unavailable: {e}"}
    token = os.environ.get("HF_TOKEN")
    for name in VOCAB_MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir, token=token)
            out[name] = vocab_scripts(tok)
            print(f"  vocab {name}: {out[name]}", flush=True)
        except Exception as e:
            out[name] = {"status": f"failed: {type(e).__name__}: {str(e)[:120]}"}
    return out


def main():
    cache = sys.argv[1] if len(sys.argv) > 1 else None
    os.makedirs(RESULTS, exist_ok=True)
    report = {"note": "CC-100 is the corpus released as XLM-R's training data. "
                      "Arabic-script lines are split by function words; see "
                      "classifier_validation for its accuracy."}

    def save():
        with open(OUT, "w") as f:
            json.dump(report, f, indent=2)

    if "validation" in SECTIONS:
        print("classifier validation", flush=True)
        report["classifier_validation"] = validate_classifier()
        print(" ", report["classifier_validation"], flush=True)
        save()
    if "wikipedia" in SECTIONS:
        print("wikipedia", flush=True)
        report["wikipedia"] = wikipedia()
        save()
    if "vocab" in SECTIONS:
        print("vocabularies", flush=True)
        report["vocabularies"] = vocabularies(cache)
        save()
    if "cc100" in SECTIONS:
        print("cc100", flush=True)
        report["cc100"] = cc100()
        save()
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
