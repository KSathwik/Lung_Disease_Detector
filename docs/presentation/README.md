# LungAI M.Tech presentation

Files
- `slides.md`: slide text, tables and speaker notes (edit this). Format is described at the top of the file.
- `build_presentation.py`: builds `LungAI_MTech_Presentation.pptx` (16:9) and then reopens it to count slides, estimate text overflow and list `[PENDING ...]` markers.
- Inputs read at build time: `docs/manuscripts/shared/results.json` (numbers, via `{{Key}}`) and `docs/figures/<stem>.png` (a labelled placeholder box is drawn when a PNG is missing).

Build
```
D:\LungAI\.venv-audit\Scripts\python.exe docs\presentation\build_presentation.py
D:\LungAI\.venv-audit\Scripts\python.exe docs\presentation\build_presentation.py --verify-only
```

Status (2026-11-22)
- Framed as a designed-and-developed system: 30 main slides, then 3 backup slides on the supplementary source analysis (thesis Appendix G) after "Thank you".
- Built from the final model v6.1; demo screenshots are real captures (`docs/figures/screenshots/`). No overflow; the only marker left is the corresponding author `[TO BE CONFIRMED]` on the publications slide.
- If a model or the results change: regenerate `results.json` and the figures, then rebuild.
