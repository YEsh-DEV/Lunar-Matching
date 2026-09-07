# LUNA-MATCH: Planetary Remote Sensing Workbench
## Front-End Architecture & Interactive Visualization Specification

**Document Type:** Front-End Technical Architecture & UX Design Blueprint  
**Target Audience:** Smart India Hackathon (SIH26166) Evaluation Panel / ISRO Planetary Scientists  
**System Designation:** LUNA-MATCH Interactive Planetary Remote Sensing Console  
**Design Philosophy:** Mission-Control Analytical Tool — High-Information-Density, Zero-Fluff, Physics-Grounded

---

## 1. Executive Architectural Vision & Evaluation Strategy

To win over ISRO evaluators and senior planetary cartographers, the frontend cannot behave like a generic consumer image-upload utility. Planetary scientists evaluate registration quality based on **photogrammetric rigor, coordinate integrity, illumination geometry, and statistical spatial uniformity**.

### What Judges Look for When Viewing the Screen:
1. **Physical Understanding of Lunar Imagery:** Clear visual awareness that lunar images lack atmospheric diffusion, possess pitch-black binary shadows, and exhibit severe scale disparities (e.g., 0.25 m/px OHRC vs 1.2 m/px LRO NAC vs 80 m/px IIRS).
2. **Interactive Proof of Alignment:** Tools that allow them to personally verify crater rim continuity across boundaries down to sub-pixel resolution ($< 0.3\text{ px}$).
3. **Statistical Honesty:** Transparent display of outlier rejection (demonstrating that shifting shadow edges were correctly discarded) and spatial uniformity (demonstrating that smooth lunar maria are not left empty).
4. **Mission-Ready Usability:** Ability to inspect solar ephemeris metadata (incidence, emission, phase angles), switch hyperspectral channels, view residual vector directions, and export standard Ground Control Point (GCP) tables.

---

## 2. High-Level Screen Layout & Information Architecture

The application adopts a **4-Pane Fixed Command Center Layout** inspired by space operations consoles (such as NASA Ames Stereo Pipeline GUI, ISRO ISSDC Data Portals, and QGIS Planetary Cartography suites).

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ TOP BAR: Problem ID: SIH26166 | Mission Mode Selector | Mission Presets | Processing Status | RUN REGISTRATION   │
├──────────────────────────┬────────────────────────────────────────────────────────┬──────────────────────────────┤
│ LEFT PANEL (25% Width)   │ CENTER CANVAS (50% Width)                              │ RIGHT PANEL (25% Width)      │
│ TELEMETRY & INGESTION    │ MULTI-MODAL PLANETARY WORKBENCH                        │ QUANTITATIVE VALIDATION      │
│                          │                                                        │                              │
│ • Dual Ingestion Cards   │ Mode Tabs:                                             │ • Real-time RMSE Gauge       │
│   (Moving vs Reference)  │ [Tie-Points] [Checkerboard] [Curtain] [Residual Heat]  │   (<0.5 px Pass Indicator)   │
│                          │                                                        │                              │
│ • Sensor Metadata &      │ • Synchronized Dual Viewport (Pan/Zoom linked)         │ • Inlier Ratio Wheel         │
│   Ephemeris Inspector    │ • Colored Vector Correspondence Lines                  │   (>75% Target)              │
│                          │ • 15×15 Micro-Zoom Loupe Window                        │                              │
│ • IIRS Band Selector     │ • Dynamic Crosshair Coordinate Display                 │ • Residual Quiver Plot       │
│                          │ • Spatial Density Quadtree Overlay                     │   (Vector Distortion Map)    │
│ • Resolution Pyramid     │                                                        │                              │
│   Scale Space Slider     │ Sub-Controls:                                          │ • Spatial Distribution Index │
│                          │ • Candidate Filter: [Raw] → [ANMS] → [Inliers]         │   (Shannon Entropy Score)    │
│ • Model Hyperparameter   │ • Checkerboard Tile Dimension Slider                   │                              │
│   Tuning Accordion       │ • Curtain Swipe Divider Handle                         │ • Export Center:             │
│                          │ • 2 Hz Flicker / Opacity Blender                       │   [GeoTIFF] [GCP CSV] [PDF]  │
└──────────────────────────┴────────────────────────────────────────────────────────┴──────────────────────────────┘
```

### Color Palette & Visual Tone:
* **Background:** Deep Cosmic Obsidian (`#0A0D14`) to mimic space data systems and minimize eye fatigue during high-contrast image analysis.
* **Surface Panels:** Dark Slate Charcoal (`#121824`) with subtle borders (`#1E293B`) for clean visual separation.
* **Inlier / Valid State:** Electric Emerald (`#10B981`) — represents verified physical ground correspondences.
* **Outlier / Shadow Rejection:** Crimson Flare (`#EF4444`) — represents discarded false shadow/parallax matches.
* **Telemetry & Structural Feature Accents:** Radiant Cyan (`#00F2FE`) and Solar Amber (`#F59E0B`).
* **Typography:** Monospaced tabular figures for all coordinates, angles, and RMSE statistics to ensure high scientific precision.

---

## 3. In-Depth Module Specifications

---

### MODULE 1: Dual Ingestion & Metadata Inspector (Left Panel)

This panel manages data loading, raster header extraction, and radiometric normalization parameters before processing begins.

#### 1. Dual File Dropzones & Auto-Detection:
* **Moving Image Slot (Chandrayaan-2):** Accepts OHRC (~0.25–0.32 m/px), TMC-2 (~5 m/px), or IIRS (~80 m/px) in GeoTIFF, PDS4, or VICAR IMG formats.
* **Reference Image Slot (NASA/SELENE):** Accepts LRO NAC (~0.5–1.5 m/px) or Kaguya TC.
* **Auto-Extracted Header Badges:** As soon as a file is dropped, the client reads and renders:
  * Sensor Name & Mission Origin
  * Ground Sampling Distance (GSD) in meters/pixel
  * Raster Dimensions (Width × Height × Bands)
  * Bit Depth & Radiometric Range (e.g., 16-bit Unsigned Integer, 32-bit Float Radiance)
  * Coordinate Reference System (CRS) & Projection (e.g., Lunar Polar Stereographic, Equirectangular)

#### 2. Solar Ephemeris & Illumination Compass:
* **Why this impresses ISRO:** Proves that the platform accounts for changing solar angles rather than treating rasters as flat terrestrial photos.
* **Visual Sun Compass:** An interactive 360° circular dial showing:
  * **Sun Azimuth Arrow:** The directional heading of incident sunlight.
  * **Sun Elevation Gauge:** The angle of the sun above the local lunar horizon.
  * **Phase Angle ($\alpha$):** The angular separation between the solar illumination vector and the spacecraft camera vector.
* **Delta Display:** Highlights the acute difference between Source and Reference sun angles (e.g., $\Delta\text{Azimuth} = 142^\circ$, $\Delta\text{Elevation} = 28^\circ$), visually explaining why classical cross-correlation fails due to inverted crater shadows.

#### 3. Hyperspectral Channel Selector (for IIRS Data):
* Displays a spectral slider across the 256 bands ($0.8\,\mu\text{m} \text{ to } 5.0\,\mu\text{m}$).
* Provides a quick-toggle for **Pseudo-Panchromatic Synthesis Mode**, showing the weighted continuum composite ($0.7–1.0\,\mu\text{m}$) used by the backend RIFT module to enable cross-modal structural registration with panchromatic OHRC/NAC.

#### 4. Dynamic Resolution Pyramid Slider:
* An interactive multi-step slider: `[Full Res (OHRC 0.28m)]` $\rightarrow$ `[Octave 1 (0.56m)]` $\rightarrow$ `[Octave 2 (1.12m)]` $\rightarrow$ `[TMC-2 Scale Space (4.48m)]`.
* Allows judges to visually observe how high-resolution images are downsampled to bridge the scale disparity before coarse transformer matching.

#### 5. Dynamic 16-bit Radiometric Range Stretch:
* Planetary rasters store raw Digital Numbers (DN) with wide dynamic range.
* The UI provides client-side histogram stretching controls (Min-Max, Linear 2%–98% percentile clip, and CLAHE) to ensure dark crater floors and bright sunlit peaks are perfectly visible without blowing out contrast.

---

### MODULE 2: Live Correspondence & Spatial Uniformity Engine (Center Canvas — View 1)

This view visualizes how the algorithm detects, filters, and validates tie-points, explicitly demonstrating that the system satisfies ISRO's spatial uniformity criteria.

#### 1. Synchronized Dual-Pane Viewport:
* Side-by-side display of Source (left) and Reference (right) with coupled panning and zooming.
* Smooth vector lines connect matching landmarks across the split boundary.

#### 2. The 3-Stage Match Hierarchy Toggle:
Judges can click between three sequential states to observe the exact pipeline filtering stages:
1. **Stage A: Raw Transformer Proposals (Blue Vectors):**
   * Visualizes the raw candidate matches generated by LoFTR / RoMa (e.g., 3,500 candidates).
   * Shows candidates penetrating into low-texture lunar plains (*maria*) where traditional SIFT/ORB find zero points.
2. **Stage B: Post-ANMS Uniform Distribution (Cyan Vectors):**
   * Visualizes the result of Adaptive Non-Maximal Suppression ($c_{\text{robust}} = 0.9$).
   * Demonstrates how points on crowded crater rims are thinned out while solitary points in low-contrast zones are preserved, ensuring an even spatial spread.
3. **Stage C: Post-MAGSAC++ Geometric Inliers (Dual-Color View):**
   * **Green Vectors:** Validated physical ground inliers that conform to the underlying terrain model.
   * **Red Vectors:** Outliers successfully rejected by $\sigma$-consensus (e.g., false matches on moving shadow contours or repetitive micro-craters).

#### 3. Interactive $15 \times 15$ Sub-Pixel Micro-Zoom Loupe:
* When a user hovers their cursor over any tie-point on either image:
  * A floating circular magnified loupe ($8\times$ zoom) appears over both the source point and the matching reference point.
  * Shows the local $15 \times 15$ pixel neighborhood with sub-pixel interpolation.
  * A high-contrast crosshair marks the exact floating-point coordinate ($x + \Delta x, y + \Delta y$), visually proving the sub-pixel precision achieved by the Lucas-Kanade gradient engine.

#### 4. Spatial Density Heatmap Overlay:
* An optional toggle that superimposes an $8 \times 8$ quadtree grid over the image.
* Each grid cell is colored by match point density (from dark blue = low to bright emerald = optimal).
* Guarantees to the evaluators that there are zero "dead zones" across the entire frame.

---

### MODULE 3: Photogrammetric Inspection & Verification Suite (Center Canvas — Views 2, 3, 4)

Once registration is executed, the user switches the central viewport to inspect the warped, aligned result against the reference image using standard planetary cartography verification tools.

#### 1. Split-Screen Curtain Slider (View 2):
* An interactive vertical divider bar that the user can drag horizontally across the screen.
* To the left of the divider is the Reference Image; to the right is the Warped Chandrayaan-2 Image.
* As the judge swipes back and forth across a large impact crater or central peak, the crater rim remains stationary, demonstrating that perspective distortion and relief displacement have been absorbed by the Thin Plate Spline (TPS) transformation.

#### 2. Dynamic Checkerboard Alignment View (View 3):
* Interleaves alternate square tiles of the Reference image and the Registered Source image in a unified grid.
* **Interactive Tile Size Slider:** Allows the user to adjust checkerboard tile dimensions from $16 \times 16\text{ px}$ to $256 \times 256\text{ px}$.
* **The "Rim Test":** Circular crater rims that traverse across adjacent checkerboard tiles must align seamlessly without step-discontinuities or sheared edges.

#### 3. 2 Hz Flicker & Continuous Opacity Blender (View 4):
* **Flicker Mode:** Alternates between the Reference and Registered Source image at 2 cycles per second ($2\text{ Hz}$). The human eye is exceptionally sensitive to motion; any residual registration error causes the feature to "jump" back and forth. A rock-solid image with zero jumping confirms sub-pixel alignment.
* **Continuous Blend Slider:** A manual $0\% \text{ to } 100\%$ alpha transparency fader for smooth cross-dissolve inspection.

#### 4. Absolute Residual Difference Heatmap (View 5):
* Calculates and renders the absolute intensity difference:
  $$\Delta I(x, y) = |I_{\text{reference}}(x, y) - I_{\text{registered}}(x, y)|$$
* Uses a scientific colormap (e.g., Viridis or Inferno).
* **The Evaluator's Litmus Test:** Under different sun angles, the floor of a crater will naturally show intensity differences due to shifted shadows. However, the physical crater boundary and rim edges show near-zero residual difference, visually proving that the model aligned physical terrain rather than illumination artifacts.

---

### MODULE 4: Quantitative Validation Dashboard (Right Panel)

This panel provides real-time mathematical validation of the four mandatory competition metrics.

#### 1. Sub-Pixel RMSE Radial Gauge:
* A semi-circular gauge displaying the final Root Mean Square Error:
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N \left( (x_i^{\text{ref}} - \hat{x}_i^{\text{src}})^2 + (y_i^{\text{ref}} - \hat{y}_i^{\text{src}})^2 \right)}$$
* Prominently displays the numerical value (e.g., **$0.27\text{ px}$**).
* Features a prominent colored status badge:
  * Green: **"SUB-PIXEL PASS (< 0.5 px)"**
  * Gold: **"ACCEPTABLE (< 0.8 px)"**
  * Red: **"DEGRADED (> 1.0 px)"**

#### 2. Inlier Ratio & Consensus Health Wheel:
* A dual-ring donut chart:
  * Outer Ring: Total Initial Proposals (e.g., 2,840 points).
  * Inner Ring: Validated Geometric Inliers (e.g., 2,315 points).
  * Center Stat: **Inlier Ratio: $81.5\%$** (exceeding ISRO's $>75\%$ target threshold).

#### 3. Residual Vector Quiver Plot (Parallax & Distortion Map):
* A 2D vector field plot displaying the residual error vectors $(\Delta x_i, \Delta y_i)$ at each inlier coordinate.
* **Vector Amplification Slider ($10\times, 50\times, 100\times$):** Since true sub-pixel errors ($0.2\text{ px}$) are microscopic on a standard display, this slider magnifies the vector arrow lengths.
* **Cartographic Interpretation for Judges:**
  * If arrows point randomly in all directions with near-zero length, registration errors are pure uncorrelated Gaussian noise (the ideal state).
  * If arrows swirl or point uniformly in one direction, systematic scale drift, rotation error, or unmodeled 3D relief parallax remains.

#### 4. Spatial Distribution Index (SDI) & Shannon Entropy Score:
* Quantifies the 2D spatial dispersion of match points across an $M \times M$ grid ($8 \times 8 = 64$ cells):
  $$\text{SDI} = -\sum_{k=1}^{K} p_k \log_2(p_k)$$
  where $p_k = \frac{n_k}{N}$ is the proportion of points located in grid cell $k$.
* Normalized Entropy readout: e.g., **$\text{SDI} = 0.94 / 1.00$** accompanied by a badge reading **"Uniform Spatial Coverage Verified"**.

#### 5. Processing Latency Breakdown:
* A stacked horizontal bar displaying the time spent in each pipeline phase:
  * Preprocessing & Lommel-Seeliger Normalization: $1.2\text{ s}$
  * Phase Congruency & MIM Generation: $3.8\text{ s}$
  * Transformer Coarse Matching: $4.5\text{ s}$
  * MAGSAC++ & TPS Verification: $1.1\text{ s}$
  * Lucas-Kanade Sub-Pixel Refinement: $1.4\text{ s}$
  * Total Runtime: **$12.0\text{ s}$** (well under the 60-second operational target).

---

### MODULE 5: Planetary Export & Ground Control Engine (Bottom of Right Panel)

Provides one-click export of analysis-ready scientific deliverables compliant with planetary science standards.

#### 1. Analysis-Ready GeoTIFF Download:
* Exports the warped, registered Chandrayaan-2 image with updated affine geotransform tags and CRS embedded in the GeoTIFF header, ready for instant drag-and-drop into QGIS or ArcGIS Pro.

#### 2. Ground Control Point (GCP) CSV Export:
* Generates a standard photogrammetric control point table formatted as:
  ```
  Point_ID, Source_X, Source_Y, Ref_X, Ref_Y, SubPixel_dx, SubPixel_dy, Residual_px, Confidence, Quad_ID
  GCP_0001, 1420.24,  891.12,   710.15, 445.60,  -0.08,        +0.12,        0.24,        0.962,      Q1_2
  GCP_0002, 3810.85, 2190.40,  1905.42, 1095.22,  +0.11,        -0.05,        0.18,        0.984,      Q3_4
  ```

#### 3. PDF Registration Audit Certificate:
* Generates a single-page formal verification report featuring:
  * Mission & Sensor Identification (e.g., *CH2_OHRC_20201115 vs LRO_NAC_M114264582R*)
  * Solar ephemeris metadata table
  * Side-by-side diagnostic thumbnails (tie-points, quiver plot, checkerboard)
  * Final statistical audit table (RMSE, Inlier Ratio, SDI, Max Residual)
  * Digital verification seal for competition records.

---

## 4. Frontend Performance Engineering for Gigapixel Rasters

Full-size Chandrayaan-2 OHRC frames can reach $12,000 \times 40,000$ pixels. Loading such massive uncompressed rasters directly into standard browser DOM elements will instantly crash browser memory.

### Frontend Architectural Solutions:
1. **Tiled Virtual Viewport (DeepZoom / Slippy Map Architecture):**
   * The center canvas utilizes dynamic multi-resolution tile streaming (via WebGL or tiled HTML5 Canvas).
   * Only the visible viewport tiles at the current zoom level are requested and rendered into GPU memory.
2. **Offscreen Canvas Rendering for Interactive Overlays:**
   * Vector tie-point lines, ANMS circles, and quiver arrows are rendered on an independent overlay canvas layer using WebGL point-sprites. Panning and zooming remain fluid at 60 FPS without re-rendering the underlying raster.
3. **Client-Side Sub-Pixel Magnifier Loupe:**
   * The $15 \times 15$ sub-pixel loupe does not perform expensive full-image transformations. It extracts a tiny $15 \times 15$ matrix centered on the target coordinate and applies a local bicubic interpolation shader directly in the client.

---

## 5. Strategic Recommendations to "WOW" ISRO Evaluators

### 1. Pre-Loaded Mission Presets (The "Failsafe Demo" Feature):
Live hackathon network connections and GPU servers can occasionally experience lag. To ensure a flawless live presentation, include a **"Mission Presets"** dropdown in the top bar with pre-computed, verified scenarios:
* **Preset 1 (Extreme Sun-Angle Shift):** *Chandrayaan-2 OHRC vs LRO NAC at Boguslawsky Crater (South Pole)* — Solar incidence angle difference $\Delta = 58^\circ$, showing how Phase Congruency and LoFTR find matches despite pitch-black shadow inversions.
* **Preset 2 (Massive Scale Disparity):** *Chandrayaan-2 TMC-2 ($5\text{ m/px}$) vs OHRC ($0.28\text{ m/px}$)* — Demonstrating resolution pyramid bridging across an $18\times$ scale gap.
* **Preset 3 (Cross-Modal Hyperspectral):** *Chandrayaan-2 IIRS ($80\text{ m/px}$, 256 bands) vs LRO NAC ($1.2\text{ m/px}$)* — Demonstrating pseudo-panchromatic continuum synthesis and RIFT MIM structural alignment.

### 2. Interactive "Why Classical SIFT Failed" Comparison Modal:
* A toggle button labeled **"Compare with SIFT / ORB"**.
* Clicking it opens a split view showing classical SIFT locking onto false shadow edges (producing $>80\%$ outliers and rim clustering) right next to LUNA-MATCH's uniform, terrain-locked inliers. Evaluators will immediately see the value of your research-backed approach.

### 3. Non-Rigid Relief Parallax Toggle (Homography vs TPS):
* Provide a toggle between **"Flat Homography"** and **"Thin Plate Spline (TPS)"**.
* When Homography is selected, show how residual error arrows flare up around steep crater walls due to 3D terrain relief displacement.
* When TPS is enabled, watch the residual arrows contract to near-zero, proving that your pipeline absorbs true 3D lunar topography.

---

## 6. Implementation Readiness Summary

The proposed front-end workbench directly maps to the capabilities developed in the 4 backend modules:
* **Module 1 (Ingestion):** Feeds the dual dropzones, metadata card, solar compass, and pyramid slider.
* **Module 2 (Matching):** Feeds the raw transformer proposals and ANMS uniform distribution viewer.
* **Module 3 (Verification):** Feeds the MAGSAC++ inlier/outlier color classification and Quiver distortion plot.
* **Module 4 (Warping & Refinement):** Feeds the Lucas-Kanade $15 \times 15$ loupe, curtain slider, dynamic checkerboard, difference heatmap, and GeoTIFF/GCP export engine.

This document serves as the complete technical architecture and user experience blueprint for building the LUNA-MATCH Planetary Remote Sensing Workbench.
