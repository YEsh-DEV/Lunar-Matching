# Chandrayaan-2 Mission Reference

## Mission Overview
ISRO Chandrayaan-2 launched July 2019. Orbiter operational at 100km polar
orbit. Carries three imaging payloads for lunar surface mapping.

## OHRC (Orbiter High Resolution Camera)
Resolution: 0.25m/px at 100km orbit. Swath: 3km.
Best for: landing site characterization, boulder mapping, small crater
detection (diameter > 1m). Single panchromatic band.
Registration challenge: small swath means limited overlap with TMC-2.
Solar angle sensitivity: high — shadows dominate at incidence > 70deg.

## TMC-2 (Terrain Mapping Camera 2)
Resolution: 5m/px. Swath: 20km. Stereo pairs (fore/aft).
Used for: DEM generation, regional geology, large crater morphology.
Scale disparity vs OHRC: 20x. Phase congruency + coarse-to-fine required.

## IIRS (Imaging Infrared Spectrometer)
Wavelength: 0.8-5.0 micron. Resolution: 80m/px.
Used for: mineralogy mapping. Rarely registered to OHRC/TMC-2 directly
due to 320x scale disparity — spectral unmixing preferred.

## LRO LROC NAC (Reference)
NASA Lunar Reconnaissance Orbiter. NAC resolution: 0.5m/px.
Used as ground-truth reference for Chandrayaan-2 OHRC validation.
Cross-mission registration: OHRC (0.25m) vs LROC NAC (0.5m) = 2x disparity.
Well within classical SIFT scale invariance range.
