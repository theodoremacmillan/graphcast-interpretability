#!/bin/bash
# -----------------------------------------------------------------------------
# GraphCast Interpretability — Sherlock HPC setup
#
# This script loads required system modules and prepares a Python virtual
# environment for installing Python dependencies that rely on system libraries
# (PROJ, GEOS, libtiff).
#
# Usage:
#   source scripts/setup_sherlock.sh
# -----------------------------------------------------------------------------

echo "Loading HPC system modules..."

module load python/3.12
module load cuda/12.6.1
module load physics
module load libtiff/4.5.0
module load proj/9.5.1
module load geos/3.13.1
python -m pip uninstall -y cartopy pyproj shapely
python -m pip install --only-binary=cartopy,pyproj,shapely cartopy pyproj shapely

echo "System modules loaded."

# -----------------------------------------------------------------------------
# Create virtual environment if it does not exist
# -----------------------------------------------------------------------------

if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment (.venv)..."
    python3.12 -m venv .venv
else
    echo "Virtual environment (.venv) already exists."
fi

# -----------------------------------------------------------------------------
# Activate virtual environment
# -----------------------------------------------------------------------------

echo "Activating virtual environment..."
source .venv/bin/activate

# -----------------------------------------------------------------------------
# Upgrade packaging tools
# -----------------------------------------------------------------------------

echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# -----------------------------------------------------------------------------
# Install geospatial dependencies (binary-only)
# -----------------------------------------------------------------------------

echo "Installing geospatial dependencies (binary-only)..."
pip install --only-binary=cartopy,pyproj,shapely \
    Cartopy pyproj shapely

# -----------------------------------------------------------------------------
# Install remaining Python dependencies
# -----------------------------------------------------------------------------

if [ -f "requirements.txt" ]; then
    echo "Installing remaining Python dependencies..."
    pip install -r requirements.txt
else
    echo "ERROR: requirements.txt not found."
    exit 1
fi

echo "Environment setup complete."
echo "Activate later with: source .venv/bin/activate"
