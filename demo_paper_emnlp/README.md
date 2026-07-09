# EMNLP 2026 System Demonstrations — Paper Source

Paper: **AutoTerm-SST: Zero-Setup Adaptive Terminology Memory for Streaming
Speech Translation**.

## Build

```bash
cd demo_paper_emnlp
pdflatex -output-directory latex latex/acl_latex.tex
(cd latex && bibtex acl_latex)
pdflatex -output-directory latex latex/acl_latex.tex
pdflatex -output-directory latex latex/acl_latex.tex
# output: latex/acl_latex.pdf
```

### Regenerate Figure 1

`latex/figures/autoterm_architecture.svg` is the editable vector master. Export
both manuscript and preview assets from that same source:

```bash
rsvg-convert -f pdf -b white \
  -o latex/figures/autoterm_architecture.pdf \
  latex/figures/autoterm_architecture.svg
rsvg-convert -b white -w 2400 \
  -o latex/figures/autoterm_architecture.png \
  latex/figures/autoterm_architecture.svg
```

## Layout

- `latex/acl_latex.tex` — main file (ACL style, `preprint` mode: EMNLP demo
  track is single-blind, so author names stay visible).
- `latex/sections/` — one file per section.
- `latex/figures/` — editable architecture source and its PDF/PNG exports, UI
  evidence panel, and routing timeline.
- `latex/custom.bib` — references.
- Table/figure sources: `../runtime/eval_20260621/paper_tables.md` and
  `../docs/` eval reports.

## Submission checklist (EMNLP 2026 demo CFP)

- [x] ≤ 6 pages main content (references may overflow; verified content ends
  on page 6)
- [x] Evaluation reported (desk-reject condition)
- [x] Licensing addressed in the paper (MIT + upstream licenses)
- [ ] Screencast video ≤ 2.5 min — **replace the placeholder URL in the
  footnote in `sections/5_demo_interface.tex`**
- [ ] Live demo link or downloadable package link — **strict requirement;
  update the same footnote**
- [ ] Fill pending main results (32-session stress numbers in
  `sections/6_evaluation.tex`) from the final run
- Camera-ready: restore the Acknowledgments block in `acl_latex.tex`
  (commented out for the submission page budget; accepted papers get +1 page).
