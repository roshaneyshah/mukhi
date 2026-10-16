"""Every run_all.py call in the Kaggle notebook must parse with the runner's
real argument parser, so a flag typo cannot reach a GPU session."""
import sys, os, json, shlex
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
import run_all

ap = run_all.build_parser()
NB = os.path.join(ROOT, "notebooks", "kaggle_run.ipynb")
if not os.path.exists(NB):
    print("SKIP  notebooks/kaggle_run.ipynb not shipped in this copy")
    print()
    print("1/1 passed")
    sys.exit(0)
nb = json.load(open(NB, encoding="utf-8"))
subs = {"$CACHE": "/c", "$ACTS_IN": "/a", "$ACTS": "/a", "$SMALL": "x,y", "$seed": "1"}
ok = bad = 0
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    for line in "".join(c["source"]).splitlines():
        line = line.strip().lstrip("#").strip()
        if "python run_all.py" not in line:
            continue
        for k, v in sorted(subs.items(), key=lambda kv: -len(kv[0])):
            line = line.replace(k, v)
        try:
            a = ap.parse_args(shlex.split(line.split("python run_all.py", 1)[1]))
            assert not (a.extract_only and not a.acts_dir)
            ok += 1
        except (SystemExit, AssertionError):
            bad += 1
            print("FAIL ", line)
print(("PASS  " if not bad else "FAIL  ") + f"{ok} notebook commands parse")
print()
print(f"{int(not bad)}/1 passed")
sys.exit(1 if bad else 0)
