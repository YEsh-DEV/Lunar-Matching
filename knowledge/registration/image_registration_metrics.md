# Lunar Image Registration Metrics Reference

## RMSE (Root Mean Square Error)
RMSE measures average reprojection error in pixels after geometric alignment.
Sub-pixel accuracy means RMSE < 0.5px. Values 0.5-2.0px are acceptable for
reconnaissance. Above 2px indicates unreliable registration.
For ISRO lunar science: RMSE < 0.5px required for crater depth profiling,
landing site safety mapping, and multi-temporal change detection.

## Inlier Ratio
Fraction of candidate matches verified as geometrically consistent by MAGSAC++.
>80% = excellent. 30-80% = robust. <30% = sparse, treat with caution.
Low inlier ratio with low total candidates means the image pair has little
shared texture — common at extreme illumination angles or large scale disparity.

## SDI (Spatial Dispersion Index)
Shannon entropy of match distribution across an 8x8 grid over the image.
SDI > 0.6 means matches are well-spread, constraining the transform globally.
SDI < 0.3 means matches cluster in one region — transform may be unreliable
outside that cluster. Critical for TPS (Thin Plate Spline) fits.

## Held-Out RMSE and Overfit Ratio
80/20 validation split: 80% of inliers fit the transform, 20% held out.
held_out_rmse measures generalization. overfit_ratio = held_out/fit_rmse.
Ratio > 1.5 is a red flag — the transform fits training points but
extrapolates poorly. TPS with sparse points commonly shows this.

## Transform Types
- homography: 8-DOF planar projective. Assumes flat surface. Condition
  number kappa < 1e4 required for numerical stability.
- affine: 6-DOF. Used when homography is ill-conditioned (kappa > 1e4).
  Recovers rotation, scale, shear, translation.
- similarity: 4-DOF. Scale + rotation + translation only. Most stable.
- thin_plate_spline: Non-rigid. Used when relief parallax check detects
  significant 3D topography (crater walls, central peaks). Exactly
  interpolates through control points — watch for high overfit ratio.

## SSIM (Structural Similarity Index)
Measures perceptual similarity between registered and reference images.
Range -1 to 1. Values > 0.5 indicate good photometric alignment.
Low SSIM with good RMSE means geometric alignment is correct but
photometric normalization (Lommel-Seeliger) was bypassed or images
have very different illumination conditions.

## Condition Number (kappa)
SVD-based measure of homography matrix stability.
kappa < 100: well-conditioned, reliable. 100-1e4: acceptable.
>1e4: ill-conditioned, pipeline falls back to affine automatically.
