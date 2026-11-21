#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
    echo "Usage: bash latex/make_arxiv.sh <out.tar.gz>" >&2
    exit 2
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir "$work/build" "$work/stage" "$work/verify"
archive="$work/source.tar.gz"

(
    cd latex
    latexmk -pdf -interaction=nonstopmode -halt-on-error -recorder -silent \
        -outdir="$work/build" acl_latex.tex
)

python3 - "$work/build" "$work/stage" <<'PY'
import re
import shutil
import sys
from pathlib import Path

build, stage = map(Path, sys.argv[1:])
paper = Path("latex").resolve()

# Consume literal regions and escaped characters before looking for comment starts.
tokens = re.compile(
    rb"""
      \\begin\{(?P<env>verbatim\*?|(?:B|L)?Verbatim\*?|SaveVerbatim|lstlisting|minted)\}
          .*?\\end\{(?P=env)\}
    | \\verb\*?(?P<delim>[^A-Za-z\s])[^\r\n]*?(?P=delim)
    | \\[^\r\n]
    | (?P<comment> ^[ \t]*%[^\r\n]*(?:\r?\n|$) | %[^\r\n]* )
    """,
    re.MULTILINE | re.DOTALL | re.VERBOSE,
)

# The recorder gives the files TeX actually opened after applying \graphicspath
# and graphics extension precedence, including inputs from nested sections.
files = {Path(name) for name in ("acl_latex.tex", "acl.sty", "acl_natbib.bst")}
for line in (build / "acl_latex.fls").read_text().splitlines():
    if line.startswith("INPUT "):
        path = (paper / line[6:]).resolve()
        if path.is_relative_to(paper):
            rel = path.relative_to(paper)
            section = rel.is_relative_to("sections") and rel.suffix == ".tex"
            figure = rel.is_relative_to("figures") and rel.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg"}
            if section or figure:
                files.add(rel)

for rel in files:
    src, dst = paper / rel, stage / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if rel.suffix == ".tex":
        clean = tokens.sub(lambda m: b"" if m["comment"] is not None else m[0], src.read_bytes())
        dst.write_bytes(clean)
    else:
        shutil.copyfile(src, dst)
shutil.copyfile(build / "acl_latex.bbl", stage / "acl_latex.bbl")
PY

(
    cd "$work/stage"
    find . -type f | sort | COPYFILE_DISABLE=1 tar -czf "$archive" -T -
)
(
    cd "$work/verify"
    tar -xzf "$archive"
    for pass in 1 2; do
        pdflatex -interaction=nonstopmode -halt-on-error acl_latex.tex > "$work/pdflatex.stdout" || {
            tail -n 50 "$work/pdflatex.stdout" >&2
            exit 1
        }
    done
    if grep -Ei 'there were undefined (references|citations)|(reference|citation) .* undefined' acl_latex.log >&2; then
        exit 1
    fi
)

pages() {
    pdfinfo "$1" | awk '/^Pages:/ { print $2 }'
}
normal_pages=$(pages "$work/build/acl_latex.pdf")
arxiv_pages=$(pages "$work/verify/acl_latex.pdf")
if [[ -z "$normal_pages" || "$normal_pages" != "$arxiv_pages" ]]; then
    echo "Page count mismatch: latexmk=$normal_pages, archive=$arxiv_pages" >&2
    exit 1
fi

mv "$archive" "$1"
printf 'Normal latexmk build: %s pages\nExtracted arXiv package: %s pages\n' "$normal_pages" "$arxiv_pages"
tar -tzf "$1"
