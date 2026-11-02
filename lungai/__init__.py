"""LungAI — canonical research package for the six-class chest X-ray classifier.

Single source of truth for the model identity, class mapping, preprocessing, dataset
checks, training and evaluation. The backend imports from here; nothing else may
redefine the class order or the preprocessing steps.
"""

__version__ = "1.0.0"
