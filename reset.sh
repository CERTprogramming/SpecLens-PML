#!/bin/bash
# ======================================================
# SpecLens-PML Reset Script
# ======================================================
#
# This script resets the repository state to run the demo
# from scratch (fresh MLOps pipeline execution).
#
# It removes:
# - Feedback examples collected from unseen runs
# - Temporary training staging directory
# - Generated datasets
# - Trained model artifacts (candidates + champion)
#
# Raw datasets (train/test/unseen) are NOT touched.
#
# Usage:
#   ./reset.sh
#
# ======================================================

set -e

echo "=== SpecLens-PML Reset Demo State ==="

# ------------------------------------------------------
# Remove feedback pool
# ------------------------------------------------------
echo "Cleaning feedback pool..."
rm -rf data/raw_feedback/*

# ------------------------------------------------------
# Remove temporary staging directory
# ------------------------------------------------------
echo "Removing temporary training staging directory..."
rm -rf data/_tmp_train/

# ------------------------------------------------------
# Remove generated datasets
# ------------------------------------------------------
echo "Removing generated dataset CSV files..."
rm -f data/datasets_train.csv
rm -f data/datasets_test.csv

# ------------------------------------------------------
# Remove trained models
# ------------------------------------------------------
echo "Removing model artifacts..."
rm -f models/logistic.pkl
rm -f models/forest.pkl
rm -f models/best_model.pkl

echo ""
echo "Reset completed successfully."
echo "You can now run a clean demo with:"
echo "   python3 demo.py"
echo "====================================="

