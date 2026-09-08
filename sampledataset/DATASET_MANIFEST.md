# Sample Dataset Manifest

**Location:** `sampledataset/`  
**Total Files:** 18 (2 PNG + 16 JPEG)  
**Generated:** 2026-09-08  
**Note:** Read-only documentation — no files moved, renamed, or deleted.

---

## File Inventory

| # | Filename | Size (bytes) | Dimensions | Mode | Format |
|---|----------|--------------|------------|------|--------|
| 1 | image.png | 320,205 | 556 × 437 | RGB | PNG |
| 2 | image copy.png | 273,703 | 555 × 442 | RGB | PNG |
| 3 | res1.jpeg | 651,816 | 1536 × 1024 | RGB | JPEG |
| 4 | res2.jpeg | 239,013 | 1437 × 877 | RGB | JPEG |
| 5 | res3.jpeg | 306,863 | 1000 × 1000 | RGB | JPEG |
| 6 | res4.jpeg | 370,344 | 1600 × 734 | RGB | JPEG |
| 7 | res5.jpeg | 695,282 | 1509 × 1042 | RGB | JPEG |
| 8 | res6.jpeg | 635,739 | 1509 × 1042 | RGB | JPEG |
| 9 | res7.jpeg | 63,380 | 437 × 300 | RGB | JPEG |
| 10 | res8.jpeg | 631,773 | 1254 × 1254 | RGB | JPEG |
| 11 | sou1.jpeg | 168,845 | 922 × 617 | RGB | JPEG |
| 12 | sou2.jpeg | 619,330 | 1600 × 975 | RGB | JPEG |
| 13 | sou3.jpeg | 583,564 | 1254 × 1254 | RGB | JPEG |
| 14 | sou4.jpeg | 462,249 | 1600 × 733 | RGB | JPEG |
| 15 | sou5.jpeg | 562,722 | 1509 × 1042 | RGB | JPEG |
| 16 | sou6.jpeg | 674,829 | 1509 × 1042 | RGB | JPEG |
| 17 | sou7.jpeg | 57,608 | 437 × 302 | RGB | JPEG |
| 18 | sou8.jpeg | 387,226 | 1100 × 1100 | RGB | JPEG |

---

## Pairing Analysis

### Clear Pairs (Dimensions Match Exactly or Within 1-2px)

| Pair | Reference (res*) | Source (sou*) | Dimensions | Match Quality | Difficulty |
|------|------------------|---------------|------------|---------------|------------|
| 4 | res4.jpeg (1600×734) | sou4.jpeg (1600×733) | 1600×734 vs 1600×733 | Height differs by 1px | **Easy** — near-identical geometry |
| 5 | res5.jpeg (1509×1042) | sou5.jpeg (1509×1042) | Exact match | Perfect | **Easy** — identical resolution |
| 6 | res6.jpeg (1509×1042) | sou6.jpeg (1509×1042) | Exact match | Perfect | **Easy** — identical resolution |
| 7 | res7.jpeg (437×300) | sou7.jpeg (437×302) | 437×300 vs 437×302 | Height differs by 2px | **Easy** — near-identical geometry |

### Probable Pairs (Similar Naming Pattern, Dimensions Differ)

| Pair | Reference (res*) | Source (sou*) | Dimensions | Notes |
|------|------------------|---------------|------------|-------|
| 1 | res1.jpeg (1536×1024) | sou1.jpeg (922×617) | Very different | May be different crops/regions |
| 2 | res2.jpeg (1437×877) | sou2.jpeg (1600×975) | Different aspect ratios | Possible different sensors |
| 3 | res3.jpeg (1000×1000) | sou3.jpeg (1254×1254) | Both square, different scale | Possible scale difference |
| 8 | res8.jpeg (1254×1254) | sou8.jpeg (1100×1100) | Both square, different scale | Possible scale difference |

### Standalone / Unclear

| File | Notes |
|------|-------|
| image.png (556×437) | Matches existing `data/samples/image_1.tif` dimensions — likely Chandrayaan-2 OHRC |
| image copy.png (555×442) | Matches existing `data/samples/image_2.tif` dimensions — likely LRO NAC reference |

**Flagged Unclear:** Pairs 1, 2, 3, 8 have significant dimension mismatches. Without metadata (GSD, sensor, acquisition date), it's unclear if these are:
- Different crops of the same region (scale difference)
- Different regions entirely
- Different sensors with different resolutions
- Misnamed/misnumbered files

---

## GSD / Geospatial Metadata

**Cannot determine from current files.** These are plain JPEG/PNG without:
- GeoTIFF affine transform (no GSD)
- CRS tags
- Solar incidence/emission/phase angle tags
- Nodata masks

**Required for LUNA-MATCH pipeline:** Convert to GeoTIFF with proper metadata (see `data/samples/REAL_DATA_README.md`).

---

## Recommended Next Steps

1. **Verify pairs 4, 5, 6, 7 first** — they have matching dimensions and will register easily
2. **Investigate pairs 1, 2, 3, 8** — check source metadata or acquisition logs to confirm pairing
3. **Convert to GeoTIFF** with GSD and solar angles before running pipeline
4. **Use existing `data/samples/verified_a.tif` / `verified_b.tif`** as ground-truth templates for metadata structure