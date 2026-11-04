"""Novelty extension (C1-C4) analysis code, implementing docs/publication/NOVELTY_PROTOCOL.md.

Order of use follows protocol section 11: features (timing, caches) -> spectral_map (C1) -> inlp_eval (C3)
-> loso (C4) -> c2_lite (C2-lite and the section 10 go/no-go) -> render_novelty (figures and macros).
TensorFlow is imported only by ``lungai.novelty.features``.
"""
