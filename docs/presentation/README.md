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

After the final model is chosen
1. Regenerate `results.json` and the figures, then rebuild.
2. Replace every red `[PENDING FINAL RESULTS: ...]` text in `slides.md` with interpretation written from the final numbers; update the speaker notes that say interpretation is pending.
3. Slide titles that say "run 1 values" (internal and per-class results) should be renamed.
4. Add demo screenshots (slide 25 is a placeholder), confirm author order (`[AUTHORS TO CONFIRM]`), and check the datasets table against `data/derived/<build>/v6_build_report.json`.
