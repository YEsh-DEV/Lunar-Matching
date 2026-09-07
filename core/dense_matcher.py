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
    max_features: int = 4000,
) -> np.ndarray:
    """
    Primary safety-net classical matcher using OpenCV SIFT with Lowe's ratio test
    and ORB fallback. Guaranteed to return (N, 5) float64 array.
    """
    if not _HAS_CV2:
        logger.error("OpenCV (cv2) not installed; cannot perform classical feature matching.")
        return np.empty((0, 5), dtype=np.float64)

    # Prepare 8-bit grayscale images for OpenCV
    def to_u8(img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint8:
            return img
        norm = (img - np.nanmin(img)) / max(np.nanmax(img) - np.nanmin(img), 1e-6)
        return (norm * 255.0).astype(np.uint8)

    u8_a = to_u8(img_a)
    u8_b = to_u8(img_b)

    matches_list = []

    # 1. Try SIFT
    try:
        sift = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.03, edgeThreshold=10)
        kp_a, des_a = sift.detectAndCompute(u8_a, None)
        kp_b, des_b = sift.detectAndCompute(u8_b, None)

        if des_a is not None and des_b is not None and len(kp_a) >= 4 and len(kp_b) >= 4:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            raw_matches = bf.knnMatch(des_a, des_b, k=2)

            for pair in raw_matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < 0.8 * n.distance:
                        # Map distance ratio to confidence score [0.0, 1.0]
                        ratio = m.distance / max(n.distance, 1e-6)
                        conf = float(np.clip(1.0 - ratio, 0.1, 1.0))
                        if conf >= (confidence_thresh * 0.5):
                            pt_a = kp_a[m.queryIdx].pt
                            pt_b = kp_b[m.trainIdx].pt
                            matches_list.append([pt_a[0], pt_a[1], pt_b[0], pt_b[1], conf])
    except Exception as e:
        logger.warning(f"SIFT feature matching encountered error: {e}")

    matcher_path_used = "none"
    if len(matches_list) >= 8:
        matcher_path_used = f"Classical SIFT with Lowe's ratio test ({len(matches_list)} correspondences)"
        logger.info(f"Matcher executed: {matcher_path_used}")

    # 2. Fallback to ORB if SIFT yielded insufficient matches
    if len(matches_list) < 8:
        logger.info("SIFT yielded < 8 matches; attempting ORB safety net.")
        try:
            orb = cv2.ORB_create(nfeatures=max_features)
            kp_a, des_a = orb.detectAndCompute(u8_a, None)
            kp_b, des_b = orb.detectAndCompute(u8_b, None)

            if des_a is not None and des_b is not None and len(kp_a) >= 4 and len(kp_b) >= 4:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                raw_matches = bf.knnMatch(des_a, des_b, k=2)
                for pair in raw_matches:
                    if len(pair) == 2:
                        m, n = pair
                        if m.distance < 0.85 * n.distance:
                            conf = float(np.clip(1.0 - (m.distance / max(n.distance, 1e-6)), 0.1, 1.0))
                            pt_a = kp_a[m.queryIdx].pt
                            pt_b = kp_b[m.trainIdx].pt
                            matches_list.append([pt_a[0], pt_a[1], pt_b[0], pt_b[1], conf])
            if len(matches_list) >= 4 and matcher_path_used == "none":
                matcher_path_used = f"Classical ORB fallback ({len(matches_list)} correspondences)"
                logger.info(f"Matcher executed: {matcher_path_used}")
        except Exception as e:
            logger.warning(f"ORB fallback matching encountered error: {e}")

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


def run_dense_matching(
    img_a: np.ndarray,
    img_b: np.ndarray,
    model: Optional[object] = None,
    confidence_thresh: float = 0.5,
) -> np.ndarray:
    """
    Execute dense/semi-dense image correspondence.

    Attempts deep LoFTR matching if available; seamlessly falls back
    to classical SIFT/ORB safety-net matcher.

    Parameters
    ----------
    img_a             : (H, W) float64 ndarray
    img_b             : (H, W) float64 ndarray
    model             : optional pre-loaded matcher model
    confidence_thresh : minimum match confidence [0.0, 1.0]

    Returns
    -------
    matches : (N, 5) float64 array of [x1, y1, x2, y2, confidence]
    """
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
