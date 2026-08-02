"""
Centralized repository path definitions.

This module provides a single source of truth for all important
directories used across the Machine Learning Platform.
"""

from pathlib import Path

# Repository root
ROOT_DIR = Path(__file__).resolve().parents[3]

# Main directories
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
DOCS_DIR = ROOT_DIR / "docs"
TESTS_DIR = ROOT_DIR / "tests"

# Data directories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Optional directories
LOGS_DIR = ROOT_DIR / "logs"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
