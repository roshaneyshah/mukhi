#!/bin/sh
# Build paper/arxiv/main.pdf from paper/draft.md (XeLaTeX, Noto fonts for
# Gurmukhi and Shahmukhi). Run from the repo root: sh paper/build_pdf.sh
set -e
cd "$(dirname "$0")/arxiv"
cp ../../results/figures/scale_depth.png . 2>/dev/null || true
python3 -I - <<'PY'
import re
s=open('../draft.md',encoding='utf-8').read()
s=s.replace('(`results/figures/scale_depth.png`)','\n\n![](scale_depth.png){width=90%}')
i=s.index('\n## Abstract')
head=s[:i]; s='# '+head.split('\n',1)[0][2:]+'\n\nCode, result files and the findings log: github.com/roshaneyshah/mukhi.\n'+s[i:]
open('main.md','w',encoding='utf-8').write(s)
PY
pandoc main.md -t latex --shift-heading-level-by=-1 -o body.tex
python3 -I - <<'PY'
import re
b=open('body.tex',encoding='utf-8').read()
b=re.sub(r'([਀-੿][਀-੿‌‍]*)', r'{\\gurfont \1}', b)
b=re.sub(r'([؀-ۿݐ-ݿ][؀-ۿݐ-ݿً-ٟ‌ ]*[؀-ۿݐ-ݿ]|[؀-ۿ])', r'{\\arabfont\\beginR \1\\endR}', b)
open('body_x.tex','w',encoding='utf-8').write(b)
PY
xelatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
xelatex -interaction=nonstopmode main.tex > build.log 2>&1 || true
grep -E "^!" build.log && exit 1 || echo "built arxiv/main.pdf"
