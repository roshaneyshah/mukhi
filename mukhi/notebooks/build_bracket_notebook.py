#!/usr/bin/env python3
"""Build mukhi_bootstrap.ipynb: the paired sentence-level bootstrap run.

One GPU session answers reviewer point W3. The paper's central quantity is
trained-minus-random, and until now it had no inferential test: the
BH-corrected p values attach to a sign-flip null for a single run. This run
re-extracts the three encoders' activations and resamples sentences, with the
trained and random-init passes resampled identically so the difference is
paired.

Because FLORES+ is parallel, one shared resample also makes the Section 5.4
ordering (Punjabi trough against Serbian trough) and the Section 5.7 scale
claim (base against large) testable, not merely reported.

The per-draw differences are saved as .npz alongside the summary, so any
further interval can be recomputed without another GPU session.
"""
import base64
import io
import json
import os
import tarfile

REPO = "/home/claude/mukhi"
OUT = "/home/claude/mukhi_bracket_decoder.ipynb"

SKIP_DIRS = {".git", "__pycache__", "acts", "data", "results_backup", "paper"}
# Kaggle caps notebook source at 1MB. The conditions (6MB raw) are rebuilt
# from Hugging Face at run time instead, and computed results stay out.
SKIP_FILES = {"main_pumvr_enc.json", "conditions_flores.json",
              "conditions_sib.json", "audit_sheet.csv"}


def keep(path):
    parts = path.split(os.sep)
    if any(p in SKIP_DIRS for p in parts):
        return False
    if os.path.basename(path) in SKIP_FILES:
        return False
    if path.endswith((".pyc", ".zip", ".npz")):
        return False
    if ".pre_" in os.path.basename(path):
        return False
    if parts[0] == "notebooks":
        return False
    if parts[0] == "results":
        if parts[-1].endswith(".png"):
            return False
        # The published main_* results are the ONE exception to the usual
        # exclusion: the run cross-checks its own point estimates against
        # them, so the intervals are known to attach to the paper's numbers.
        # Only the three encoders are needed, and only their paired blocks,
        # so they are slimmed below rather than shipped whole.
        if parts[-1].startswith(("intervention_", "fertility_")):
            return False
        if parts[-1].startswith("main_") and not _is_encoder_main(parts[-1]):
            return False
    return True


ENCODERS = ("xlm-roberta-base", "xlm-roberta-large", "muril-base-cased")


def _is_encoder_main(fname):
    return fname.startswith("main_flores_") and any(e in fname for e in ENCODERS)


def slim_main(path):
    """Keep only what the cross-check reads: per-layer paired consistency.

    The full files carry CKA curves, probe results and permutation baselines,
    several MB in total. The cross-check needs one float per layer per
    contrast, so shipping the rest would spend the 1MB budget on nothing.
    """
    d = json.load(open(path))
    out = {"model": d.get("model"), "pooling": d.get("pooling"),
           "n": d.get("n"), "random_init": d.get("random_init"),
           "seed": d.get("seed"), "contrasts": {}}
    for k, v in (d.get("contrasts") or {}).items():
        paired = v.get("paired")
        if paired:
            out["contrasts"][k] = {
                "paired": [{"layer": r.get("layer"),
                            "consistency": r.get("consistency")} for r in paired]}
    return json.dumps(out, separators=(",", ":")).encode()


def build_tar():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as t:
        for root, dirs, files in os.walk(REPO):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for f in sorted(files):
                full = os.path.join(root, f)
                rel = os.path.relpath(full, REPO)
                if not keep(rel):
                    continue
                if os.path.basename(rel).startswith("main_flores_"):
                    data = slim_main(full)
                    info = tarfile.TarInfo("mukhi/" + rel)
                    info.size = len(data)
                    info.mode = 0o644
                    # A hand-built TarInfo defaults to mtime=0, which extracts
                    # as 1970. zipfile refuses timestamps before 1980, so the
                    # first zip of results/ raised and took the whole run with
                    # it 100 seconds in.
                    info.mtime = int(os.path.getmtime(full))
                    t.addfile(info, io.BytesIO(data))
                else:
                    t.add(full, arcname="mukhi/" + rel)
    return buf.getvalue()


def literal(name, b64):
    lines = [b64[i:i + 128] for i in range(0, len(b64), 128)]
    return name + " = (\n" + "\n".join('    "%s"' % l for l in lines) + "\n)\n"


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


UNPACK = '''
import base64, io, os, tarfile, json
tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD))).extractall(
    "/kaggle/working" if os.path.isdir("/kaggle/working") else ".")
ROOT = "/kaggle/working/mukhi" if os.path.isdir("/kaggle/working") else "./mukhi"
CACHE = "/tmp/hf_cache"   # not /kaggle/working, which has a 20GB cap
os.makedirs(CACHE, exist_ok=True)
os.chdir(ROOT)
pub = sorted(f for f in os.listdir("results") if f.startswith("main_flores_"))
print("working in", os.getcwd())
print("published results shipped for cross-check:", len(pub))
for f in pub:
    print("  ", f)
'''

TOKEN = '''
SECRET_LABEL = "kaggle-mukhi"

!pip -q install uroman 2>/dev/null
import os, torch

tok = None
try:
    from kaggle_secrets import UserSecretsClient
    tok = UserSecretsClient().get_secret(SECRET_LABEL)
except Exception as e:
    print("Could not read the Kaggle secret:", str(e)[:160])

if not tok:
    raise SystemExit(
        "No Hugging Face token.\\n"
        "Secrets are attached per notebook, so a new notebook starts with "
        "none even though others have it.\\n"
        "Fix: Add-ons > Secrets, switch on '" + SECRET_LABEL + "' for THIS "
        "notebook, then re-run.")

os.environ["HF_TOKEN"] = tok
os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
try:
    from huggingface_hub import login
    login(token=tok, add_to_git_credential=False)
except Exception as e:
    print("login() failed, environment variables still set:", str(e)[:100])

# Preflight. dataset_info answers for gated repos WITHOUT a token, so it is
# useless as an auth check and gave a false pass on an earlier run.
# Downloading a file from the gated repo does require authorisation.
from huggingface_hub import HfApi, hf_hub_download
who = HfApi().whoami(token=tok)
print("authenticated as:", who.get("name", "?"))
hf_hub_download("openlanguagedata/flores_plus", "README.md",
                repo_type="dataset", token=tok)
print("gated FLORES+ download authorised")

print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| gpus", torch.cuda.device_count())
assert torch.cuda.is_available(), "Turn the GPU on: Settings > Accelerator > GPU T4 x2"
'''

SELFTEST = '''
# The bootstrap's whole method rests on one algebraic shortcut: that resampled
# consistency can be read off weighted sums instead of recomputed per draw.
# Verify that here, in this session, before spending GPU hours on it.
import subprocess, sys
for t in ("tests/test_bootstrap.py", "tests/test_bootstrap_stage.py",
          "tests/test_bracket.py"):
    p = subprocess.run([sys.executable, t], capture_output=True, text=True)
    tail = (p.stdout or "").strip().splitlines()[-1:] or ["(no output)"]
    print(("ok   " if p.returncode == 0 else "FAIL "), t, "|", tail[0])
    if p.returncode != 0:
        print((p.stdout or "")[-2500:], (p.stderr or "")[-1500:])
        raise SystemExit("bootstrap self-test failed; not spending GPU time")
print("\\nself-tests pass")
'''

RUN = '''
# Press once and leave. Models run one at a time, weights are purged after
# each, the zip is refreshed after EVERY model, and a failure is recorded
# rather than killing the run.
import subprocess, sys, time, shutil, os

log = []


def run(label, args, script="run_all.py"):
    t0 = time.time()
    print("\\n" + "=" * 72 + "\\n" + label, flush=True)
    p = subprocess.run([sys.executable, script] + args,
                       capture_output=True, text=True)
    out = (p.stdout or "")[-9000:] + (p.stderr or "")[-3000:]
    print(out, flush=True)
    log.append((label, p.returncode, round(time.time() - t0)))
    return p.returncode == 0


EPOCH_1980 = 315532800


def zip_now(why):
    # Any file whose mtime predates 1980 makes zipfile raise, and that once
    # ended this run at its first zip. Normalise first, and never let a
    # failure here throw away hours of finished work.
    try:
        now = time.time()
        for root, _dirs, files in os.walk("results"):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    if os.path.getmtime(fp) < EPOCH_1980:
                        os.utime(fp, (now, now))
                except OSError:
                    pass
        shutil.make_archive("/kaggle/working/mukhi_bracket_results", "zip", "results")
        kb = os.path.getsize("/kaggle/working/mukhi_bracket_results.zip") // 1024
        print(f"  zip refreshed after {why}: {kb} KB", flush=True)
    except Exception as e:
        print(f"  zip after {why} FAILED ({type(e).__name__}: {str(e)[:120]}); "
              f"results/ on disk is intact and the run continues", flush=True)


# Conditions come from Hugging Face. Without them nothing downstream can run,
# so a failure here skips the models rather than burning GPU hours on nothing.
have = run("conditions", ["--stage", "conditions", "--cache-dir", CACHE])
if have:
    zip_now("conditions")

SEEDS = "0,1,2"
ORDER = ["xlm-roberta-base", "xlm-roberta-large", "google/muril-base-cased"]
BRACKET = "punjabi,script_form,script_form_dv1,script_form_dv2,script_form_rt"

# Part 1: the bracket test. Only the Punjabi contrasts and the four controls
# are needed, all in one extraction group, on the same resample seed as the
# published run.
for m in (ORDER if have else []):
    run("bracket " + m, [
        "--stage", "bootstrap", "--models", m, "--boot-only", BRACKET,
        "--n-boot", "2000", "--boot-seed", "12345",
        "--boot-random-seeds", SEEDS, "--max-length", "512",
        "--cache-dir", CACHE, "--purge-cache", "--force",
        "--out-tag", "bracket_" + m.split("/")[-1]])
    zip_now(m)

if have:
    run("bracket check (from saved draws, no GPU)", [],
        script="tools_bracket_check.py")
    zip_now("bracket check")

# Part 2: one decoder, mean-pooled (Finding 8 excluded last-token pooling),
# with three random-init seeds and every contrast. Decoder tokenisers spend
# up to 1.8 tokens per Gurmukhi character, so the length limit is raised;
# the stage stops if anything is still truncated.
if have:
    run("decoder Qwen2.5-1.5B, mean-pooled", [
        "--stage", "bootstrap", "--models", "Qwen/Qwen2.5-1.5B",
        "--pooling", "mean",
        "--n-boot", "2000", "--boot-seed", "12345",
        "--boot-random-seeds", SEEDS, "--max-length", "2048",
        "--batch-size", "4",
        "--cache-dir", CACHE, "--purge-cache", "--force",
        "--out-tag", "decoder_meanpool"])
    zip_now("decoder")

print("\\n" + "=" * 72)
for label, rc, secs in log:
    print(("ok  " if rc == 0 else "FAIL") + "  %-46s %6ds" % (label, secs))
print("\\ntotal %ds" % sum(s for _, _, s in log))

zip_now("end of run")
print("\\nDownload mukhi_bracket_results.zip from the Output panel.")
'''

TAR = build_tar()

cells = [
    md("""
# MUKHI, bracket test and decoder

One run, two parts.

1. **Bracket test.** Two new controls sit between de-voweled and
   round-tripped Gurmukhi. If the script-form share falls steadily from
   de-voweled to round-tripped, the 66 to 90% range holds. The result is
   written to `results/bracket_check_flores.json`.
2. **One decoder.** Qwen2.5-1.5B, mean-pooled, three random-init seeds,
   every contrast.

Roughly 60 to 90 minutes. The zip is refreshed after every step.

Set **Settings > Accelerator > GPU T4 x2**, switch on the `kaggle-mukhi`
secret under **Add-ons > Secrets** for this notebook, then **Save Version >
Save & Run All (Commit)**.
"""),
    md("## 1. Unpack"),
    code(literal("PAYLOAD", base64.b64encode(TAR).decode()) + UNPACK),
    md("""
## 2. Hugging Face token

Needs the Kaggle secret `kaggle-mukhi` to hold a **Read** token on an account
that has accepted the FLORES+ terms. Secrets attach **per notebook**: switch
it on under **Add-ons > Secrets** for this notebook even if other notebooks
already have it. This cell stops the run if the token is missing or
unauthorised, rather than failing six hours later.
"""),
    code(TOKEN),
    md("""
## 3. Self-test, before spending GPU time

The method rests on one algebraic shortcut. These tests check it against a
naive per-draw recomputation and exercise the whole stage on synthetic
activations with a known planted structure. Seconds to run, and they fail
closed.
"""),
    code(SELFTEST),
    md("""
## 4. The run
"""),
    code(RUN),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3",
                                  "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

def smoke_test(tar_bytes):
    """Unpack the payload and zip results/, exactly as the run cell will.

    The first run died at its first zip because ten hand-built tar members
    carried mtime=0, which zipfile rejects. Reproducing that costs a second,
    so it belongs here rather than 100 seconds into a GPU session.
    """
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp()
    try:
        tarfile.open(fileobj=io.BytesIO(tar_bytes)).extractall(tmp)
        root = os.path.join(tmp, "mukhi")
        bad = []
        for dirpath, _d, fs in os.walk(root):
            for f in fs:
                fp = os.path.join(dirpath, f)
                if os.path.getmtime(fp) < 315532800:
                    bad.append(os.path.relpath(fp, root))
        if bad:
            raise SystemExit(
                "ERROR: %d payload file(s) carry pre-1980 mtimes, which "
                "zipfile rejects: %s" % (len(bad), ", ".join(sorted(bad)[:4])))
        shutil.make_archive(os.path.join(tmp, "z"), "zip",
                            os.path.join(root, "results"))
        print("smoke test: payload unpacks and results/ zips cleanly")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


smoke_test(TAR)

json.dump(nb, open(OUT, "w"))
size = os.path.getsize(OUT)
print("wrote %s  %d KB  %d cells" % (OUT, size // 1024, len(cells)))
if size > 1_000_000:
    raise SystemExit("ERROR: %d bytes exceeds Kaggle's 1MB kernel source cap" % size)
