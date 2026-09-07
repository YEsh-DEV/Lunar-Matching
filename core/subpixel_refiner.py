"""
core/subpixel_refiner.py  (v2 — LK sign fix + bilinear fix)
"""
import numpy as np
import logging
logger = logging.getLogger(__name__)

try:
    from scipy.ndimage import map_coordinates
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _sample_patch_bilinear(image: np.ndarray, cx: float, cy: float, half_size: int):
    H, W = image.shape
    pw   = 2 * half_size + 1
    if cx < half_size or cx >= W - half_size or cy < half_size or cy >= H - half_size:
        return None

    if _HAS_SCIPY:
        rows = np.arange(pw, dtype=np.float64) - half_size + cy
        cols = np.arange(pw, dtype=np.float64) - half_size + cx
        rr, cc = np.meshgrid(rows, cols, indexing='ij')
        patch  = map_coordinates(image, [rr.ravel(), cc.ravel()], order=1, mode='nearest').reshape(pw, pw)
    else:
        r0 = max(0, int(round(cy)) - half_size)
        c0 = max(0, int(round(cx)) - half_size)
        patch = image[r0:r0+pw, c0:c0+pw].copy()
        if patch.shape != (pw, pw):
            return None

    return patch.astype(np.float64)


def _image_gradients(patch: np.ndarray) -> tuple:
    Ix = np.zeros_like(patch)
    Iy = np.zeros_like(patch)
    Ix[:, 1:-1] = 0.5 * (patch[:, 2:] - patch[:, :-2])
    Ix[:,    0] = patch[:,  1] - patch[:,  0]
    Ix[:,   -1] = patch[:, -1] - patch[:, -2]
    Iy[1:-1, :] = 0.5 * (patch[2:, :] - patch[:-2, :])
    Iy[  0,  :] = patch[1,  :] - patch[0,  :]
    Iy[ -1,  :] = patch[-1, :] - patch[-2, :]
    return Ix, Iy


def lucas_kanade_refine(
    img_f: np.ndarray,
    img_g: np.ndarray,
    point: tuple,
    patch_size: int = 15,
    iterations: int = 8,
    min_eigenvalue: float = 1e-6,
    convergence_threshold: float = 0.03,
    max_shift: float = 1.0,
    return_info: bool = False,
) -> tuple:
    """
    Refine correspondence via Lucas-Kanade gradient descent.

    Normal equations:
      [Σ Ix²    Σ Ix*Iy]   [Δx]   [Σ Ix*(G-F)]
      [Σ Ix*Iy  Σ Iy²  ] · [Δy] = [Σ Iy*(G-F)]

    We hold F fixed (template from img_f at point (x_a, y_a)) and
    iteratively shift the sampling position in img_g by (Δx, Δy).

    Returns (x_b_refined, y_b_refined, converged) or
    (x_b_refined, y_b_refined, converged, info_dict) if return_info=True.
    """
    x_a, y_a, x_b, y_b = float(point[0]), float(point[1]), float(point[2]), float(point[3])
    half = patch_size // 2

    info = {
        'iterations': 0,
        'delta_mag': 0.0,
        'total_shift': 0.0,
        'reason': 'uninitialized',
        'min_eig': 0.0,
    }

    # Fixed reference template from image A
    patch_f = _sample_patch_bilinear(img_f, x_a, y_a, half)
    if patch_f is None:
        info['reason'] = 'boundary_template'
        return (x_b, y_b, False, info) if return_info else (x_b, y_b, False)

    # Gradients computed from reference template (constant across iterations)
    Ix, Iy = _image_gradients(patch_f)
    Ix_flat = Ix.ravel()
    Iy_flat = Iy.ravel()

    # Build 2×2 Hessian (structure tensor)
    H11 = float(np.dot(Ix_flat, Ix_flat))
    H12 = float(np.dot(Ix_flat, Iy_flat))
    H22 = float(np.dot(Iy_flat, Iy_flat))

    det = H11 * H22 - H12 * H12
    # Minimum eigenvalue check
    trace = H11 + H22
    disc  = max((H11 - H22) ** 2 + 4 * H12 ** 2, 0.0)
    min_eig = 0.5 * (trace - np.sqrt(disc))
    info['min_eig'] = float(min_eig)

    if min_eig < min_eigenvalue or abs(det) < 1e-12:
        info['reason'] = 'low_eigenvalue'
        return (x_b, y_b, False, info) if return_info else (x_b, y_b, False)

    # Precompute H^{-1}
    H_inv = np.array([[H22, -H12], [-H12, H11]], dtype=np.float64) / det

    cx, cy = x_b, y_b
    converged_by_threshold = False

    for it in range(iterations):
        info['iterations'] = it + 1
        patch_g = _sample_patch_bilinear(img_g, cx, cy, half)
        if patch_g is None:
            info['reason'] = 'boundary_target'
            return (x_b, y_b, False, info) if return_info else (x_b, y_b, False)

        # Error: G(cx,cy) - F(x_a,y_a)
        diff = (patch_g - patch_f).ravel()

        # Right-hand side: b = [Σ Ix*(G-F), Σ Iy*(G-F)]
        b1 = float(np.dot(Ix_flat, diff))
        b2 = float(np.dot(Iy_flat, diff))

        # LK update: displacement to apply to sampling position in G
        delta_x = H_inv[0, 0] * b1 + H_inv[0, 1] * b2
        delta_y = H_inv[1, 0] * b1 + H_inv[1, 1] * b2

        cx -= delta_x
        cy -= delta_y

        step_mag = float(np.hypot(delta_x, delta_y))
        shift_mag = float(np.hypot(cx - x_b, cy - y_b))
        info['delta_mag'] = step_mag
        info['total_shift'] = shift_mag

        # Divergence guard: subpixel refinement must not drift beyond max_shift
        if shift_mag > max_shift:
            info['reason'] = 'exceeded_max_shift'
            return (x_b, y_b, False, info) if return_info else (x_b, y_b, False)

        if step_mag < convergence_threshold:
            converged_by_threshold = True
            info['reason'] = 'converged_threshold'
            break

    if not converged_by_threshold and info['reason'] == 'uninitialized':
        info['reason'] = 'iteration_cap'

    return (cx, cy, True, info) if return_info else (cx, cy, True)


def refine_all_matches(
    img_a: np.ndarray,
    img_b: np.ndarray,
    matches: np.ndarray,
    patch_size: int = 15,
    iterations: int = 8,
    max_shift: float = 1.0,
) -> np.ndarray:
    if matches.ndim != 2 or matches.shape[1] != 5:
        raise ValueError(f"matches must be (N, 5); got {matches.shape}")

    def _norm(img):
        img = img.astype(np.float64)
        mn, mx = img.min(), img.max()
        return (img - mn) / (mx - mn) if mx > mn else img

    img_a_n = _norm(img_a)
    img_b_n = _norm(img_b)

    refined = matches.copy()
    n_ok = n_fail = 0

    counts = {
        'converged_threshold': 0,
        'iteration_cap': 0,
        'exceeded_max_shift': 0,
        'low_eigenvalue': 0,
        'boundary': 0,
    }
    non_converged_samples = []

    for i in range(len(matches)):
        x1, y1, x2, y2, conf = matches[i]
        x2r, y2r, ok, info = lucas_kanade_refine(
            img_a_n, img_b_n, (x1, y1, x2, y2),
            patch_size=patch_size, iterations=iterations,
            max_shift=max_shift,
            return_info=True,
        )
        refined[i, 2] = x2r
        refined[i, 3] = y2r
        if ok:
            n_ok += 1
            if info['reason'] == 'converged_threshold':
                counts['converged_threshold'] += 1
            else:
                counts['iteration_cap'] += 1
        else:
            n_fail += 1
            reason = info['reason']
            if reason in ('boundary_template', 'boundary_target'):
                counts['boundary'] += 1
            elif reason == 'low_eigenvalue':
                counts['low_eigenvalue'] += 1
            elif reason == 'exceeded_max_shift':
                counts['exceeded_max_shift'] += 1
            else:
                counts['exceeded_max_shift'] += 1

            if len(non_converged_samples) < 10:
                non_converged_samples.append({
                    'idx': i,
                    'iterations': info['iterations'],
                    'delta_mag': info['delta_mag'],
                    'total_shift': info['total_shift'],
                    'reason': info['reason'],
                })

    logger.info(f"LK refinement: {n_ok}/{len(matches)} converged, {n_fail} kept original.")
    logger.info(
        f"LK breakdown: {counts['converged_threshold']} hit threshold (<0.03px), "
        f"{counts['iteration_cap']} reached iter cap, "
        f"{counts['exceeded_max_shift']} exceeded max_shift ({max_shift}px), "
        f"{counts['low_eigenvalue']} low eigenvalue, {counts['boundary']} boundary."
    )
    if non_converged_samples:
        samples_detail = "; ".join(
            f"pt#{s['idx']}: iter={s['iterations']}, final_delta={s['delta_mag']:.4f}px, "
            f"shift={s['total_shift']:.4f}px, reason={s['reason']}"
            for s in non_converged_samples[:5]
        )
        logger.info(f"LK non-converged sample (first {min(5, len(non_converged_samples))}): {samples_detail}")

    return refined
