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
OUT = "/home/claude/mukhi_round3.ipynb"

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
          "tests/test_bracket.py", "tests/test_resource_count.py"):
    p = subprocess.run([sys.executable, t], capture_output=True, text=True)
    tail = (p.stdout or "").strip().splitlines()[-1:] or ["(no output)"]
    print(("ok   " if p.returncode == 0 else "FAIL "), t, "|", tail[0])
    if p.returncode != 0:
        print((p.stdout or "")[-2500:], (p.stderr or "")[-1500:])
        raise SystemExit("bootstrap self-test failed; not spending GPU time")
print("\\nself-tests pass")
'''

RUN = '''
# Press once and leave. Three queues run at the same time: GPU 0, GPU 1 and
# a CPU job. Each queue runs its steps in order and a failed step is logged,
# not fatal. The zip is refreshed after every step.
import subprocess, sys, time, shutil, os, threading

log, lock = [], threading.Lock()
EPOCH_1980 = 315532800


def zip_now(why):
    with lock:
        try:
            now = time.time()
            for root, _d, files in os.walk("results"):
                for f in files:
                    fp = os.path.join(root, f)
                    try:
                        if os.path.getmtime(fp) < EPOCH_1980:
                            os.utime(fp, (now, now))
                    except OSError:
                        pass
            shutil.make_archive("/kaggle/working/mukhi_round3_results", "zip", "results")
            kb = os.path.getsize("/kaggle/working/mukhi_round3_results.zip") // 1024
            print(f"  [zip] after {why}: {kb} KB", flush=True)
        except Exception as e:
            print(f"  [zip] after {why} FAILED ({type(e).__name__}); run continues", flush=True)


def step(queue, label, args, script="run_all.py", gpu=None):
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.makedirs("results/logs", exist_ok=True)
    lf = "results/logs/%s.log" % label.replace(" ", "_").replace("/", "_")
    t0 = time.time()
    print(f"[{queue}] start {label}", flush=True)
    with open(lf, "w") as fh:
        p = subprocess.run([sys.executable, script] + args, stdout=fh,
                           stderr=subprocess.STDOUT, env=env)
    secs = round(time.time() - t0)
    with lock:
        log.append((queue, label, p.returncode, secs))
    print(f"[{queue}] {'ok' if p.returncode == 0 else 'FAIL'} {label} ({secs}s)", flush=True)
    tail = open(lf).read()[-2500:]
    print(tail, flush=True)
    zip_now(label)
    return p.returncode == 0


BOOT = ["--n-boot", "2000", "--boot-seed", "12345", "--boot-random-seeds", "0,1,2",
        "--cache-dir", CACHE, "--purge-cache", "--force"]
ENC = ["xlm-roberta-base", "xlm-roberta-large", "google/muril-base-cased"]
PUMVR_ONLY = ("punjabi,script_form,script_form_rt,script_form_dv1,script_form_dv2,"
              "information,punjabi_gen,gen_vs_human")
DEC_ONLY = "punjabi,serbian,no_script"


def gpu0():
    # 1. Every encoder, every contrast, with per-seed draws saved.
    for m in ENC:
        step("gpu0", "flores " + m, ["--stage", "bootstrap", "--models", m,
             "--max-length", "512", "--out-tag", "round3_" + m.split("/")[-1]] + BOOT, gpu=0)
    step("gpu0", "bracket check flores", ["flores"], script="tools_bracket_check.py")
    # 2. Human-written PuMVR with random-init baselines, for the first time.
    for m in ENC:
        step("gpu0", "pumvr " + m, ["--stage", "bootstrap", "--data", "pumvr",
             "--models", m, "--boot-only", PUMVR_ONLY, "--max-length", "512",
             "--out-tag", "round3_" + m.split("/")[-1]] + BOOT, gpu=0)
    step("gpu0", "bracket check pumvr", ["pumvr"], script="tools_bracket_check.py")
    # 3. A second decoder, mean-pooled.
    step("gpu0", "decoder Qwen2.5-3B", ["--stage", "bootstrap", "--models", "Qwen/Qwen2.5-3B",
         "--pooling", "mean", "--boot-only", DEC_ONLY, "--max-length", "2048",
         "--batch-size", "4", "--out-tag", "Qwen2.5-3B"] + BOOT, gpu=0)


def gpu1():
    # A third decoder, mean-pooled, in fp32 (Gemma-2 overflows in fp16 on T4).
    step("gpu1", "decoder gemma-2-2b", ["--stage", "bootstrap", "--models", "google/gemma-2-2b",
         "--pooling", "mean", "--boot-only", DEC_ONLY, "--max-length", "2048",
         "--batch-size", "2", "--out-tag", "gemma-2-2b"] + BOOT, gpu=1)
    # A third XLM-R size for the scale comparison.
    step("gpu1", "flores xlm-roberta-xl", ["--stage", "bootstrap", "--models",
         "facebook/xlm-roberta-xl", "--max-length", "512", "--batch-size", "8",
         "--out-tag", "round3_xlm-roberta-xl"] + BOOT, gpu=1)


def cpu():
    step("cpu", "resource counts", [CACHE], script="tools_resource_count.py")


# Shared inputs first, before anything runs in parallel.
have = step("setup", "conditions", ["--stage", "conditions", "--cache-dir", CACHE])
sys.path.insert(0, "src")
from pumvr import fetch_pumvr
print("PuMVR at", fetch_pumvr(os.path.join(os.getcwd(), "data", "pumvr")), flush=True)

threads = [threading.Thread(target=f) for f in (gpu0, gpu1, cpu)] if have else []
for t in threads:
    t.start()
t0 = time.time()
while any(t.is_alive() for t in threads):
    time.sleep(600)
    with lock:
        done = len(log)
    print(f"[heartbeat] {round((time.time() - t0) / 60)} min, {done} steps finished", flush=True)
for t in threads:
    t.join()

# After both GPUs are free: the scale comparison across three sizes (CPU,
# from saved draws), then the behavioural arm with a 7B decoder split over
# both GPUs.
step("final", "scale comparison", [], script="tools_bootstrap_crossmodel.py")
step("final", "behavioural Qwen2.5-7B", ["--stage", "intervention_mcq", "--models",
     "Qwen/Qwen2.5-7B", "--cache-dir", CACHE, "--purge-cache",
     "--out-tag", "qwen2.5-7b"])

print("\\n" + "=" * 72)
for q, label, rc, secs in log:
    print(("ok  " if rc == 0 else "FAIL") + "  %-6s %-34s %6ds" % (q, label, secs))
print("\\nwall %ds" % (time.time() - t0))
zip_now("end of run")
print("\\nDownload mukhi_round3_results.zip from the Output panel.")
'''

TAR = build_tar()

cells = [
    md("""
# MUKHI, round 3

One run, three queues at once (GPU 0, GPU 1 and CPU).

1. **Encoders, every contrast.** All eleven contrasts with intervals, and
   per-seed draws so the control ordering can be checked seed by seed.
2. **PuMVR with random-init baselines.** The human-written set, now on the
   same trained-minus-random basis as FLORES.
3. **Two more decoders, mean-pooled.** Qwen2.5-3B and Gemma-2-2B.
4. **A third XLM-R size.** XLM-R XL, for the scale comparison.
5. **Generated against human Shahmukhi** on PuMVR, same sentences.
6. **The behavioural arm** with Qwen2.5-7B.
7. **Resource counts.** Script shares in CC-100 (XLM-R's training data),
   Wikipedia sizes, and tokeniser vocabularies.

Roughly 6 hours. The zip is refreshed after every step.

Set **Settings > Accelerator > GPU T4 x2** and **Internet on**, switch on the
`kaggle-mukhi` secret under **Add-ons > Secrets** for this notebook, then
**Save Version > Save & Run All (Commit)**. The Hugging Face account needs
access to google/gemma-2-2b.
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
