# Montgomery vs Rahman TB overlap check (R1): INCONCLUSIVE

Run 2026-11-03 ~04:11 local, `scripts/data/check_external_overlap.py` (MD5, then 64-bit pHash
within 10 bits, verified by thumbnail correlation >= 0.97).

- Montgomery (138) vs Rahman TB Kaggle release (4,200): **0 matches** (`external_overlap.json`).
- Positive control, Shenzhen (662) vs the same archive: **also 0 matches**
  (`../overlap_control_shenzhen_rahman_tb/external_overlap.json`), although the Rahman TB
  release is described as including NLM (Shenzhen/Montgomery) images.

The control did not detect the expected overlap, so the method most likely misses
Rahman's re-processed copies (resizing/cropping). The Montgomery result therefore does **not**
show the absence of a leak. R1 stays **unresolved** for the historical V5 model.

V6 is unaffected: it does not use the Rahman TB archive, and its build checked that no
Montgomery file appears in the development data (`external_md5_in_development: 0`).
