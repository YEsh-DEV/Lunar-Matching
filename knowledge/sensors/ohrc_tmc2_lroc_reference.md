# Sensor Specifications for Cross-Sensor Registration

## Scale Disparity Table
| Source | Reference | GSD Ratio | Pipeline Mode |
|--------|-----------|-----------|---------------|
| OHRC 0.25m | LROC NAC 0.5m | 2x | classical, fast |
| OHRC 0.25m | TMC-2 5m | 20x | classical, standard |
| TMC-2 5m | LROC WAC 100m | 20x | classical, standard |
| IIRS 80m | TMC-2 5m | 16x | not recommended |

## Illumination Conditions
Lommel-Seeliger photometric normalization requires solar incidence angle (i)
and emission angle (e) from image metadata. If absent, normalization is
bypassed and a warning is added to quality_assessment.warnings.
Typical lunar incidence angles: 0-85 degrees. At i > 70deg, shadows cover
>30% of crater floors — SIFT features will be sparse in shadowed regions.

## Crater Size Context
OHRC detectable: craters > 1m diameter. TMC-2: > 15m. LROC NAC: > 2m.
Crater density relates to surface age (size-frequency distribution).
Power-law slope of SFD histogram indicates relative age: steeper slope
= older, more saturated surface.
