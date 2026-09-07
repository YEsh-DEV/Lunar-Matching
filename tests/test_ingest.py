"""
tests/test_ingest.py
====================
Unit and integration tests for core/ingest_preprocess.py.
"""

import sys
import os
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.ingest_preprocess import (
    RasterMetadata,
    read_raster,
    lommel_seeliger_normalize,
    build_octave_pyramid,
    align_gsd,
    _HAS_RASTERIO,
)


def test_raster_metadata_interface():
    """RasterMetadata must support both dataclass attribute access and dict .items() access."""
    meta = RasterMetadata(
        gsd=0.25,
        incidence_angle=60.0,
        emission_angle=10.0,
        phase_angle=55.0,
        crs="EPSG:30100",
        shape=(1024, 1024),
    )
    # Attribute access
    assert meta.gsd == 0.25
    assert meta.incidence_angle == 60.0
    assert meta.crs == "EPSG:30100"

    # Dict-like access (required by orchestrator.py: {kk: str(vv) for kk, vv in v.items()})
    items = dict(meta.items())
    assert items['gsd'] == 0.25
    assert items['incidence_angle'] == 60.0
    assert meta['gsd'] == 0.25
    assert meta.get('crs') == "EPSG:30100"


def test_read_raster_npy():
    """read_raster must successfully ingest .npy format buffers."""
    rng = np.random.default_rng(42)
    synthetic = rng.uniform(10, 200, (64, 64)).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        npy_path = f.name
        np.save(npy_path, synthetic)

    try:
        img, meta = read_raster(npy_path)
        assert img.shape == (64, 64)
        assert img.dtype == np.float64
        # Range should be normalized to [0, 1]
        assert img.min() >= 0.0
        assert img.max() <= 1.0
        assert meta.shape == (64, 64)
        assert meta.gsd > 0
    finally:
        if os.path.exists(npy_path):
            os.remove(npy_path)


@pytest.mark.skipif(not _HAS_RASTERIO, reason="rasterio not installed")
def test_read_raster_geotiff():
    """read_raster must extract GSD, CRS, and sun angles from GeoTIFF headers."""
    import rasterio
    from rasterio.transform import from_origin

    data = np.ones((64, 64), dtype=np.float32) * 0.5
    transform = from_origin(0.0, 64.0, 0.5, 0.5)

    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        tif_path = f.name

    try:
        with rasterio.open(
            tif_path, 'w',
            driver='GTiff',
            height=64, width=64,
            count=1, dtype='float32',
            crs='EPSG:4326',
            transform=transform,
        ) as dst:
            dst.write(data, 1)
            dst.update_tags(
                INCIDENCE_ANGLE="50.0",
                EMISSION_ANGLE="8.0",
                PHASE_ANGLE="45.0",
            )

        img, meta = read_raster(tif_path)
        assert img.shape == (64, 64)
        assert abs(meta.gsd - 0.5) < 1e-4
        assert abs(meta.incidence_angle - 50.0) < 1e-4
        assert abs(meta.emission_angle - 8.0) < 1e-4
        assert abs(meta.phase_angle - 45.0) < 1e-4
    finally:
        if os.path.exists(tif_path):
            os.remove(tif_path)


def test_lommel_seeliger_normalize():
    """lommel_seeliger_normalize corrects illumination without crashing near the terminator."""
    rng = np.random.default_rng(101)
    img = rng.uniform(0.1, 0.9, (128, 128))

    meta = RasterMetadata(
        incidence_angle=89.0,  # Near terminator grazing illumination
        emission_angle=2.0,
        phase_angle=88.0,
    )
    norm = lommel_seeliger_normalize(img, meta, c1=0.5, c2=0.1)

    assert norm.shape == (128, 128)
    assert not np.isnan(norm).any()
    assert not np.isinf(norm).any()
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0


def test_build_octave_pyramid():
    """build_octave_pyramid generates correct number of scaled octave levels."""
    img = np.ones((128, 128), dtype=np.float64)
    # GSD ratio 2.0 / 0.5 = 4.0 -> ceil(log2(4)) = 2 octaves + original = 3 levels
    pyramid = build_octave_pyramid(img, source_gsd=0.5, target_gsd=2.0)

    assert len(pyramid) == 3
    assert pyramid[0].shape == (128, 128)
    assert pyramid[1].shape[0] in (64, 65)
    assert pyramid[2].shape[0] in (32, 33)


def test_align_gsd_orchestrator_convention():
    """align_gsd preserves resolution when ratio <= 5x, downsamples when extreme or forced."""
    img_a = np.ones((100, 100), dtype=np.float64)
    img_b = np.ones((50, 50), dtype=np.float64)

    meta_a = RasterMetadata(gsd=0.25)
    meta_b = RasterMetadata(gsd=0.50)

    # 1. By default, scale ratio 2.0x <= 5.0x preserves resolution
    preserved_a, preserved_b = align_gsd(img_a, img_b, meta_a, meta_b)
    assert preserved_a.shape == (100, 100)
    assert preserved_b.shape == (50, 50)

    # 2. When forced, downsampling is applied
    aligned_a, aligned_b = align_gsd(img_a, img_b, meta_a, meta_b, force=True)
    assert aligned_a.shape == (50, 50)
    assert aligned_b.shape == (50, 50)

    # 3. Extreme ratio (> 5x) with dimension floor: 1000x1000 downsampled with min_dim_floor=300
    large_a = np.ones((1000, 1000), dtype=np.float64)
    meta_large_a = RasterMetadata(gsd=0.1)
    meta_coarse_b = RasterMetadata(gsd=2.0) # 20x ratio
    down_a, _ = align_gsd(large_a, img_b, meta_large_a, meta_coarse_b, min_dim_floor=300)
    # 20x downsample would have been 50x50, but floor enforces 300px
    assert min(down_a.shape) == 300


def test_align_gsd_standalone_convention():
    """align_gsd(source_img, source_meta, ref_meta) preserves <=5x and downsamples when forced."""
    img = np.ones((100, 100), dtype=np.float64)
    meta_src = RasterMetadata(gsd=0.25)
    meta_ref = RasterMetadata(gsd=0.50)

    # Preserved by default
    preserved = align_gsd(img, meta_src, meta_ref)
    assert preserved.shape == (100, 100)

    # Downsampled when forced
    aligned = align_gsd(img, meta_src, meta_ref, force=True)
    assert aligned.shape == (50, 50)
