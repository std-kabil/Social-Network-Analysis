# IEEE LaTeX report

## Files

- `report_tex/report.tex`

  - Main IEEEtran document that uses `\input{...}` for each section.

- `report_tex/parts/*.tex`

  - Individual section files (Abstract, Introduction, Related Works, Methods, Results, Conclusion, References).

- `report_tex/concat_report.py`
  - Generates `report_tex/report_full.tex` by expanding `\input{...}` statements.

## Figures

This report uses figures from `../outputs/` via:

```latex
\graphicspath{{../outputs/}}
```

Current included figure(s):

- `network_summary_mutual.png`

Recommended (optional) additions:

- Add screenshots exported from Plotly HTML files (degree distribution, path length distribution, ROC/PR curves) as PNGs in `outputs/`, then include them using `\includegraphics`.

## Build

From the repository root:

- Compile the modular version:

  - `pdflatex report_tex/report.tex`
  - (run twice if references/labels need resolving)

- Generate and compile the single-file version:
  - `python3 report_tex/concat_report.py`
  - `pdflatex report_tex/report_full.tex`

If you prefer BibTeX instead of `thebibliography`, tell me and I can switch the reference section to a `.bib` workflow.
