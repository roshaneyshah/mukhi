"""House style: no em dashes anywhere in the repository's text, code or
notebooks (standing rule from the author)."""
import os, sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
EM = chr(0x2014)
bad = []
for dp, dn, fn in os.walk(ROOT):
    dn[:] = [d for d in dn if d not in (".git", "__pycache__", "data", "hf_cache")]
    for f in fn:
        # prose and code only: .json/.csv here are corpus text (FLORES, SIB,
        # PuMVR), whose punctuation is the source's, not ours
        if f.endswith((".py", ".md", ".ipynb", ".txt")):
            p = os.path.join(dp, f)
            try:
                for i, line in enumerate(open(p, encoding="utf-8"), 1):
                    if EM in line:
                        bad.append(f"{os.path.relpath(p, ROOT)}:{i}")
            except UnicodeDecodeError:
                pass
for b in bad[:40]:
    print("FAIL  em dash at", b)
print(f"{'PASS' if not bad else 'FAIL'}  no em dashes ({len(bad)} found)")
print()
print(f"{int(not bad)}/1 passed")
sys.exit(1 if bad else 0)
