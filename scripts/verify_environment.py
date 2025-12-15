"""
Verify GraphCast Interpretability environment.
"""

import sys

def banner(msg):
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80)

def check(condition, msg):
    if not condition:
        raise RuntimeError(f"FAILED: {msg}")
    print(f"OK: {msg}")

banner("Environment verification")

# -----------------------------------------------------------------------------
# Python
# -----------------------------------------------------------------------------
print("Python executable:", sys.executable)
print("Python version:", sys.version)
check(sys.version_info >= (3, 10), "Python >= 3.10")

# -----------------------------------------------------------------------------
# Core scientific stack
# -----------------------------------------------------------------------------
banner("Core scientific stack")

import numpy
import jax
import xarray
import graphcast

print("NumPy:", numpy.__version__)
print("JAX:", jax.__version__)
print("xarray:", xarray.__version__)
check(graphcast is not None, "GraphCast importable")

# -----------------------------------------------------------------------------
# Geospatial stack
# -----------------------------------------------------------------------------
banner("Geospatial stack")

import pyproj
import shapely
import cartopy

print("pyproj:", pyproj.__version__)
print("shapely:", shapely.__version__)
print("cartopy:", cartopy.__version__)

# -----------------------------------------------------------------------------
# PROJ linkage check (critical)
# -----------------------------------------------------------------------------
banner("PROJ linkage")

from pyproj import CRS
crs = CRS.from_epsg(4326)
check(crs is not None, "CRS.from_epsg(4326) works")

print("pyproj.show_versions():")
pyproj.show_versions()

# -----------------------------------------------------------------------------
# Cartopy CRS sanity check
# -----------------------------------------------------------------------------
banner("Cartopy CRS check")

import cartopy.crs as ccrs
crs = ccrs.PlateCarree()
check(crs is not None, "Cartopy PlateCarree CRS constructed")

# -----------------------------------------------------------------------------
# Final result
# -----------------------------------------------------------------------------
banner("SUCCESS")
print("✅ Environment appears correctly installed and linked.")
