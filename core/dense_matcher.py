"""
core/dense_matcher.py
=====================
Dense & Semi-Dense Feature Matching Module for Planetary Remote Sensing.

Implements:
  - Classical robust fallback (SIFT + Lowe's ratio test + ORB safety net)
  - LoFTR (Local Feature TRansformer) detector-free matching via kornia
  - RoMa (Robust Dense Feature Matching) interface stub
  - Automatic matcher selection based on GSD scale disparity

Interface contract:
  Returns (N, 5) float64 array: [x1, y1, x2, y2, confidence]

References:
  Lowe, D. G. (2004). Distinctive Image Features from Scale-Invariant Keypoints.
  Sun, J. et al. (2021). LoFTR: Detector-Free Local Feature Matching with Transformers.
  Edstedt, J. et al. (2023). RoMa: Robust Dense Feature Matching.
"""

import logging
from typing import Optional, Union, Tuple
import numpy as np

from core.phase_congruency_mim import extract_structural_features, mim_descriptor

logger = logging.getLogger(__name__)

# Try importing OpenCV
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    logger.warning("OpenCV not installed; dense_matcher will use synthetic fallback.")

# Try importing PyTorch & Kornia
try:
    import torch
    import kornia
    from kornia.feature import LoFTR
    _HAS_KORNIA = True
except ImportError:
    _HAS_KORNIA = False
    logger.warning("PyTorch/Kornia not installed; LoFTR will fall back to classical SIFT.")


def load_loftr_model(weights_path: Optional[str] = None, device: str = 'cpu') -> Optional[object]:
    """
    Load pre-trained LoFTR model via Kornia.

    Parameters
    ----------
    weights_path : optional local checkpoint path
    device       : 'cpu' or 'cuda'

    Returns
    -------
    model : LoFTR instance or None if loading fails
    """
    if not _HAS_KORNIA:
        logger.info("Kornia not available; LoFTR model cannot be loaded.")
        return None

    try:
        if weights_path is not None:
            matcher = LoFTR(pretrained=None)
            state_dict = torch.load(weights_path, map_location=device)
            matcher.load_state_dict(state_dict)
        else:
            matcher = LoFTR(pretrained='outdoor')

        matcher = matcher.to(device).eval()
        logger.info(f"LoFTR model loaded successfully on device: {device}")
        return matcher
    except Exception as e:
        logger.warning(f"Failed to load LoFTR weights ({e}); falling back to SIFT/ORB.")
        return None


def load_roma_model(weights_path: Optional[str] = None, device: str = 'cuda') -> object:
    """
    Load RoMa (Robust Dense Feature Matching) model.
    Stub implementation per specification for hackathon runtime.
    """
    raise NotImplementedError(
        "RoMa model is not available in hackathon demo mode; using LoFTR / classical SIFT fallback."
    )


def select_matcher(gsd_ratio: float) -> str:
    """
    Select optimal matcher architecture based on Ground Sampling Distance ratio:
      - Extreme ratio (> 3.0 or < 0.33): RoMa / multi-scale pyramid SIFT
      - Moderate ratio: LoFTR / SIFT

    Parameters
    ----------
    gsd_ratio : source_gsd / ref_gsd

    Returns
    -------
    matcher_name : 'loftr', 'roma', or 'sift'
    """
    if gsd_ratio > 3.0 or gsd_ratio < (1.0 / 3.0):
        return 'sift'  # Classical multi-scale handles large scale differences robustly
    return 'loftr'


def _run_classical_matching(
    img_a: np.ndarray,
    img_b: np.ndarray,
    confidence_thresh: float = 0.5,
) -> np.ndarray:
    """
    Classical feature matching: SIFT with Lowe's ratio test, falling back to ORB.
    Internal worker called by run_dense_matching.

    Returns (N, 5) float64 array: [x1, y1, x2, y2, confidence]
    """
    if not _HAS_CV2:
        logger.error("OpenCV (cv2) not installed; cannot perform feature matching.")
        return np.empty((0, 5), dtype=np.float64)

    def to_u8(img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint8:
            return img
        norm = (img - np.nanmin(img)) / max(np.nanmax(img) - np.nanmin(img), 1e-6)
        return (norm * 255.0).astype(np.uint8)

    u8_a = to_u8(img_a)
    u8_b = to_u8(img_b)

    matches_list = []

    # 1. Primary: SIFT + Lowe's ratio test
    try:
        sift = cv2.SIFT_create(contrastThreshold=0.02, edgeThreshold=10)
        kp_a, des_a = sift.detectAndCompute(u8_a, None)
        kp_b, des_b = sift.detectAndCompute(u8_b, None)

        if des_a is not None and des_b is not None and len(kp_a) >= 4 and len(kp_b) >= 4:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            raw_matches = bf.knnMatch(des_a, des_b, k=2)

            for pair in raw_matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < 0.80 * n.distance:
                        conf = float(np.clip(1.0 - m.distance / max(n.distance, 1e-6), 0.2, 1.0))
                        if conf >= confidence_thresh:
                            pt_a = kp_a[m.queryIdx].pt
                            pt_b = kp_b[m.trainIdx].pt
                            matches_list.append([pt_a[0], pt_a[1], pt_b[0], pt_b[1], conf])

            logger.info(f"Matcher executed: Classical SIFT with Lowe's ratio test ({len(matches_list)} correspondences)")
    except Exception as e:
        logger.warning(f"SIFT matching failed: {e}; falling back to ORB.")

    # 2. Fallback: ORB safety net if SIFT returned too few matches
    if len(matches_list) < 8:
        try:
            orb = cv2.ORB_create(nfeatures=2000, fastThreshold=12)
            kp_a, des_a = orb.detectAndCompute(u8_a, None)
            kp_b, des_b = orb.detectAndCompute(u8_b, None)

            if des_a is not None and des_b is not None and len(kp_a) >= 4 and len(kp_b) >= 4:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                raw_matches = bf.match(des_a, des_b)

                orb_matches = []
                for m in raw_matches:
                    conf = float(np.clip(1.0 - m.distance / 100.0, 0.2, 1.0))
                    if conf >= confidence_thresh:
                        pt_a = kp_a[m.queryIdx].pt
                        pt_b = kp_b[m.trainIdx].pt
                        orb_matches.append([pt_a[0], pt_a[1], pt_b[0], pt_b[1], conf])

                if len(orb_matches) > len(matches_list):
                    matches_list = orb_matches
                    logger.info(f"Matcher executed: ORB safety net ({len(matches_list)} correspondences)")
        except Exception as e:
            logger.error(f"ORB fallback also failed: {e}")

    if len(matches_list) < 8:
        logger.warning(
            f"Classical matchers found only {len(matches_list)} correspondences (minimum required is 8). "
            "No synthetic fallback will be generated."
        )

    if not matches_list:
        return np.empty((0, 5), dtype=np.float64)

    matches_arr = np.array(matches_list, dtype=np.float64)
    if matches_arr.ndim != 2 or matches_arr.shape[1] != 5:
        matches_arr = matches_arr.reshape(-1, 5)

    return matches_arr


def run_structural_matching(
    img_a: np.ndarray,
    img_b: np.ndarray,
    confidence_thresh: float = 0.5,
    max_features: int = 4000,
    ratio_thresh: float = 0.95,
) -> np.ndarray:
    """
    RIFT-style structural matching using Phase Congruency & Maximum Index Map (MIM).
    Detects keypoints, evaluates local MIM orientation histograms, and matches
    descriptors via Brute-Force L2 with Lowe's ratio test.

    Parameters
    ----------
    img_a             : (H, W) ndarray
    img_b             : (H, W) ndarray
    confidence_thresh : minimum match confidence threshold [0.0, 1.0]
    max_features      : maximum keypoints to detect per image
    ratio_thresh      : Lowe's ratio test threshold (default: 0.95)

    Returns
    -------
    matches : (N, 5) float64 array of [x1, y1, x2, y2, confidence]
    """
    if not _HAS_CV2:
        logger.error("OpenCV (cv2) not installed; cannot perform structural feature matching.")
        return np.empty((0, 5), dtype=np.float64)

    def to_u8(img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint8:
            return img
        norm = (img - np.nanmin(img)) / max(np.nanmax(img) - np.nanmin(img), 1e-6)
        return (norm * 255.0).astype(np.uint8)

    u8_a = to_u8(img_a)
    u8_b = to_u8(img_b)

    # 1. Compute Phase Congruency and Maximum Index Map (MIM)
    try:
        feats_a = extract_structural_features(img_a)
        feats_b = extract_structural_features(img_b)
        mim_a = feats_a['mim']
        mim_b = feats_b['mim']
    except Exception as e:
        logger.error(f"Failed to extract structural features for MIM matching: {e}")
        return np.empty((0, 5), dtype=np.float64)

    # 2. Keypoint detection (SIFT detector + Phase Congruency corner detector)
    kp_a, kp_b = [], []
    try:
        sift = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.02, edgeThreshold=10)
        kp_a = list(sift.detect(u8_a, None))
        kp_b = list(sift.detect(u8_b, None))
    except Exception as e:
        logger.warning(f"SIFT keypoint detection failed: {e}")

    # Augment with FAST detector on phase congruency corner map (RIFT structural keypoints)
    try:
        corner_u8_a = np.clip(feats_a['corner_map'] / max(feats_a['corner_map'].max(), 1e-6) * 255, 0, 255).astype(np.uint8)
        corner_u8_b = np.clip(feats_b['corner_map'] / max(feats_b['corner_map'].max(), 1e-6) * 255, 0, 255).astype(np.uint8)
        fast = cv2.FastFeatureDetector_create(threshold=15)
        kp_fast_a = list(fast.detect(corner_u8_a, None))
        kp_fast_b = list(fast.detect(corner_u8_b, None))
        kp_a.extend(kp_fast_a)
        kp_b.extend(kp_fast_b)
    except Exception as e:
        logger.warning(f"FAST corner detection failed: {e}")

    if len(kp_a) < 4 or len(kp_b) < 4:
        logger.warning(
            f"Structural matcher found insufficient keypoints (A: {len(kp_a)}, B: {len(kp_b)}). "
            "Returning empty matches."
        )
        return np.empty((0, 5), dtype=np.float64)

    # 3. Extract MIM descriptors at keypoint locations
    pts_a = np.array([[kp.pt[0], kp.pt[1]] for kp in kp_a], dtype=np.float32)
    pts_b = np.array([[kp.pt[0], kp.pt[1]] for kp in kp_b], dtype=np.float32)

    des_a = mim_descriptor(mim_a, pts_a)
    des_b = mim_descriptor(mim_b, pts_b)

    # 4. Match descriptors with Brute-Force L2 and ratio test
    matches_list = []
    try:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        raw_matches = bf.knnMatch(des_a, des_b, k=2)

        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < ratio_thresh * n.distance:
                    ratio = m.distance / max(n.distance, 1e-6)
                    conf = float(np.clip(1.0 - (ratio / max(ratio_thresh, 1e-6)) * 0.85, 0.15, 1.0))
                    pt_a = kp_a[m.queryIdx].pt
                    pt_b = kp_b[m.trainIdx].pt
                    matches_list.append([pt_a[0], pt_a[1], pt_b[0], pt_b[1], conf])
    except Exception as e:
        logger.warning(f"Structural matching descriptor comparison error: {e}")

    logger.info(f"Structural MIM matching executed: {len(matches_list)} raw correspondences found.")

    if not matches_list:
        return np.empty((0, 5), dtype=np.float64)

    matches_arr = np.array(matches_list, dtype=np.float64)
    if matches_arr.ndim != 2 or matches_arr.shape[1] != 5:
        matches_arr = matches_arr.reshape(-1, 5)

    return matches_arr


def run_dense_matching(
    img_a: np.ndarray,
    img_b: np.ndarray,
    gsd_ratio: float = 1.0,
    model: Optional[object] = None,
    confidence_thresh: float = 0.5,
    method: str = 'classical',
) -> np.ndarray:
    """
    Execute dense/semi-dense image correspondence.

    Dispatches to:
      - 'structural': RIFT-style Phase Congruency / MIM structural matching
      - 'classical' : Classical SIFT with Lowe's ratio test (and LoFTR if model provided)

    Parameters
    ----------
    img_a             : (H, W) float64 ndarray
    img_b             : (H, W) float64 ndarray
    model             : optional pre-loaded matcher model (LoFTR)
    confidence_thresh : minimum match confidence [0.0, 1.0]
    method            : 'classical' or 'structural' (default: 'classical')

    Returns
    -------
    matches : (N, 5) float64 array of [x1, y1, x2, y2, confidence]
    """
    if method.lower() == 'structural':
        return run_structural_matching(img_a, img_b, confidence_thresh=confidence_thresh)

    # Try LoFTR if model provided or Kornia available
    if model is not None and _HAS_KORNIA:
        try:
            device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cpu'
            t_a = torch.from_numpy(img_a).float().unsqueeze(0).unsqueeze(0).to(device)
            t_b = torch.from_numpy(img_b).float().unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                input_dict = {"image0": t_a, "image1": t_b}
                pred = model(input_dict)

            kpts0 = pred['keypoints0'].cpu().numpy()
            kpts1 = pred['keypoints1'].cpu().numpy()
            conf = pred['confidence'].cpu().numpy()

            mask = conf >= confidence_thresh
            if np.sum(mask) >= 8:
                pts0 = kpts0[mask]
                pts1 = kpts1[mask]
                scores = conf[mask]
                matches = np.column_stack([pts0[:, 0], pts0[:, 1], pts1[:, 0], pts1[:, 1], scores])
                logger.info(f"Matcher executed: LoFTR ({len(matches)} correspondences)")
                return matches.astype(np.float64)
        except Exception as e:
            logger.warning(f"LoFTR inference failed ({e}); falling back to classical SIFT.")

    # Primary reliable classical path
    return _run_classical_matching(img_a, img_b, confidence_thresh=confidence_thresh)
