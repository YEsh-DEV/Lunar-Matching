# Real Data Drop-In Convention for LUNA-MATCH

When real ISRO Chandrayaan-2 (OHRC, TMC-2, IIRS) and NASA LRO NAC / SELENE imagery arrives, place files here following this exact structure so the pipeline reads them without code changes.

---

## Expected Folder Structure

```
data/samples/
├── REAL_DATA_README.md           # This file
├── chandrayaan2/                 # Chandrayaan-2 source imagery
│   ├── ohrc/                     # OHRC panchromatic (0.25–0.3 m/px)
│   │   ├── <mission_id>_<orbit>_<tile>.tif
│   │   └── <mission_id>_<orbit>_<tile>.xml   # PDS4 label (optional, for angles)
│   ├── tmc2/                     # TMC-2 stereo (5 m/px)
│   │   ├── <mission_id>_<orbit>_<tile>_fore.tif
│   │   ├── <mission_id>_<orbit>_<tile>_aft.tif
│   │   └── <mission_id>_<orbit>_<tile>.xml
│   └── iirs/                     # IIRS hyperspectral (80 m/px, 256 bands)
│       ├── <mission_id>_<orbit>_<tile>.tif
│       └── <mission_id>_<orbit>_<tile>.xml
├── lro/                          # LRO NAC reference imagery
│   ├── nac/                      # NAC narrow-angle camera (0.5 m/px)
│   │   ├── M<orbit>_<tile>.tif
│   │   └── M<orbit>_<tile>.xml   # PDS4 label
│   └── wac/                      # WAC wide-angle (100 m/px, optional)
│       └── ...
├── selene/                       # SELENE (Kaguya) reference (optional)
│   └── ...
└── verified_a.tif                # Existing verified test pairs (keep)
    verified_b.tif
    stress_a.tif
    stress_b.tif
    ...
```

---

## Required GeoTIFF Metadata (per file)

Each `.tif` **must** contain the following tags for the pipeline to work correctly:

| Tag / Source | Required | Purpose |
|--------------|----------|---------|
| **Affine transform** (`transform.a`, `transform.e`) | ✅ Yes | GSD extraction (pixel width/height in meters) |
| **CRS** (`crs`) | ✅ Yes | Coordinate reference system (e.g., `EPSG:4326`, Moon-specific) |
| **INCIDENCE_ANGLE** (TIFF tag or PDS4 label) | ⚠️ Recommended | Solar incidence angle (degrees) for photometric normalization |
| **EMISSION_ANGLE** (TIFF tag or PDS4 label) | ⚠️ Recommended | Sensor emission angle (degrees) |
| **PHASE_ANGLE** (TIFF tag or PDS4 label) | ⚠️ Recommended | Solar phase angle (degrees) |
| **Nodata value** | ✅ Yes | Mask invalid pixels (set via GDAL `SetNoDataValue`) |

**If angles are missing:** Pipeline logs a warning and skips Lommel-Seeliger normalization (still runs but with lower accuracy).

---

## Minimal Valid GeoTIFF (Python Snippet)

```python
import rasterio
from rasterio.transform import from_origin
import numpy as np

# Example: Create a minimal valid GeoTIFF for testing
data = np.random.rand(1024, 1024).astype(np.float32)
gsd = 0.5  # meters/pixel
transform = from_origin(0.0, 0.0, gsd, gsd)

with rasterio.open(
    "data/samples/lro/nac/M123456789_tile.tif", "w",
    driver="GTiff",
    height=1024, width=1024,
    count=1, dtype="float32",
    crs="EPSG:4326",
    transform=transform,
) as dst:
    dst.write(data, 1)
    # Add solar angles as tags (degrees, float)
    dst.update_tags(
        INCIDENCE_ANGLE="45.2",
        EMISSION_ANGLE="3.1",
        PHASE_ANGLE="42.8",
    )
    dst.nodata = -9999.0
```

---

## How Pipeline Uses These Files

| Step | File | What It Reads |
|------|------|---------------|
| `core/ingest_preprocess.read_raster()` | Any `.tif` | Image array + GSD from `transform.a`, CRS, solar angles from tags |
| `core/ingest_preprocess.lommel_seeliger_normalize()` | Same | Uses `incidence_angle`, `emission_angle`, `phase_angle` |
| `core/ingest_preprocess.align_gsd()` | Pair | Computes GSD ratio, builds octave pyramid if ratio > 5× |
| `pipeline/orchestrator.py` | Pair | Expects two paths passed to `LunaMatchPipeline(job_id, img_a_path, img_b_path)` |

---

## Quick Test After Dropping Real Data

```bash
# 1. Verify rasterio can read it with angles
python -c "
import rasterio
with rasterio.open('data/samples/chandrayaan2/ohrc/YOUR_FILE.tif') as src:
    print('Shape:', src.shape)
    print('CRS:', src.crs)
    print('GSD:', abs(src.transform.a))
    print('Tags:', src.tags())
"

# 2. Run pipeline on a Chandrayaan-2 + LRO pair
python demo.py \
  --img-a data/samples/chandrayaan2/ohrc/CH2_OHRC_2023_001.tif \
  --img-b data/samples/lro/nac/M123456789_tile.tif \
  --out demo_output/real_ch2_lro
```

---

## Notes for Data Providers

- **Do not** reproject or resample — pipeline handles GSD alignment internally
- **Do not** apply any contrast enhancement — pipeline applies CLAHE + photometric normalization
- **Preserve original PDS4 labels** (`.xml`) alongside `.tif` if available; they contain precise solar geometry
- **Nodata masking** is critical for crater rims / image borders — set correctly in GeoTIFF
- **Multi-band IIRS**: pipeline currently uses band synthesis (single pseudo-panchromatic); full hyperspectral matching is a stretch goal

---

## Existing Test Pairs (Reference)

| File | GSD | Purpose |
|------|-----|---------|
| `verified_a.tif` / `verified_b.tif` | 131.6 / 219.3 m/px | Ground-truth rigid transform (8° rot, 0.6× scale) |
| `stress_a.tif` / `stress_b.tif` | 131.6 / 219.3 m/px | Harsh illumination (gamma=0.35, shadow inversion) |
| `image_1.tif` / `image_2.tif` | 131.6 / 555.6 m/px | Real lunar screenshots (4.2× scale gap) |

Use these as templates for expected GeoTIFF structure.