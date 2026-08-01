#!/usr/bin/env bash
# Export notebook to Python script for review (optional)
set -euo pipefail
jupyter nbconvert --to script notebooks/train_tinystories.ipynb --output-dir build/
