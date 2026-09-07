"""
core/phase_congruency_mim.py  (v2 - DC fix, correct ifftshift convention)
"""
import numpy as np
from numpy.fft import fft2, ifft2, ifftshift, fftshift


def log_gabor_filter_bank(
    shape: tuple,
    n_scales: int = 3,
    n_orientations: int = 6,
    min_wavelength: float = 6.0,
    mult: float = 2.1,
    sigma_on_f: float = 0.55,
    d_theta_on_sigma: float = 1.5,
) -> list:
    rows, cols = shape
    # Build frequency coordinates with DC at centre (use fftshift of fftfreq grid)
    u = np.fft.fftfreq(cols)   # DC at index 0
    v = np.fft.fftfreq(rows)
    U, V = np.meshgrid(u, v)

    # Shift so DC is at centre for angular maths, then shift back
    U_c = fftshift(U)
    V_c = fftshift(V)

    radius_c = np.sqrt(U_c**2 + V_c**2)
    theta_c  = np.arctan2(-V_c, U_c)

    # Avoid log(0) at DC — will be zeroed after
    radius_c[rows // 2, cols // 2] = 1.0

    sigma_theta = np.pi / (n_orientations * d_theta_on_sigma)

    filter_bank = []
    for s in range(n_scales):
        scale_filters = []
        wavelength_s = min_wavelength * (mult ** s)
        f_0          = 1.0 / wavelength_s
        sigma_rho    = np.log(sigma_on_f)

        radial_c = np.exp(-(np.log(radius_c / f_0) ** 2) / (2.0 * sigma_rho ** 2))
        # Zero DC at centre
        radial_c[rows // 2, cols // 2] = 0.0

        for o in range(n_orientations):
            theta_o = o * np.pi / n_orientations
            d_theta = theta_c - theta_o
            d_theta = np.where(d_theta >  np.pi / 2, d_theta - np.pi, d_theta)
            d_theta = np.where(d_theta < -np.pi / 2, d_theta + np.pi, d_theta)

            angular = np.exp(-(d_theta ** 2) / (2.0 * sigma_theta ** 2))
            filt_c  = radial_c * angular

            # Shift back to FFT convention (DC at corner [0,0])
            filt = ifftshift(filt_c)

            scale_filters.append(filt)
        filter_bank.append(scale_filters)

    return filter_bank


def compute_even_odd_responses(image: np.ndarray, filter_bank: list) -> tuple:
    img_f = fft2(image.astype(np.float64))
    n_scales       = len(filter_bank)
    n_orientations = len(filter_bank[0])
    H, W           = image.shape

    even_resp = np.zeros((n_scales, n_orientations, H, W), dtype=np.float64)
    odd_resp  = np.zeros((n_scales, n_orientations, H, W), dtype=np.float64)

    for s in range(n_scales):
        for o in range(n_orientations):
            response        = ifft2(img_f * filter_bank[s][o])
            even_resp[s, o] = response.real
            odd_resp [s, o] = response.imag

    return even_resp, odd_resp


def phase_congruency(
    image: np.ndarray,
    n_scales: int = 3,
    n_orientations: int = 6,
    noise_threshold: float = None,
    noise_std_factor: float = 3.0,
    epsilon: float = 1e-5,
    min_wavelength: float = 6.0,
    mult: float = 2.1,
    sigma_on_f: float = 0.55,
    d_theta_on_sigma: float = 1.5,
) -> np.ndarray:
    img = image.astype(np.float64)
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())

    H, W = img.shape
    fb   = log_gabor_filter_bank(
        (H, W), n_scales, n_orientations, min_wavelength, mult, sigma_on_f, d_theta_on_sigma
    )
    even_resp, odd_resp = compute_even_odd_responses(img, fb)

    numerator   = np.zeros((H, W), dtype=np.float64)
    denominator = np.zeros((H, W), dtype=np.float64)

    for o in range(n_orientations):
        F     = even_resp[:, o, :, :].sum(axis=0)
        H_vec = odd_resp [:, o, :, :].sum(axis=0)
        E_total = np.sqrt(F ** 2 + H_vec ** 2) + epsilon
        phi_e = F     / E_total
        phi_o = H_vec / E_total

        T = noise_threshold
        for s in range(n_scales):
            e_so = even_resp[s, o, :, :]
            o_so = odd_resp [s, o, :, :]
            A_so = np.sqrt(e_so ** 2 + o_so ** 2)

            dot   = e_so * phi_e + o_so * phi_o
            cross = np.abs(e_so * phi_o - o_so * phi_e)
            sharp = dot - cross

            if T is None and s == 0:
                noise_power = np.median(np.abs(A_so)) / 0.6745
                T = noise_std_factor * noise_power

            T_use = T if T is not None else 0.0
            W_o = 1.0 / (1.0 + np.exp(3.0 * (0.35 - A_so / (E_total + epsilon))))

            numerator   += np.maximum(W_o * sharp - T_use, 0.0)
            denominator += A_so

    return np.clip(numerator / (denominator + epsilon), 0.0, 1.0)


def _phase_congruency_per_orientation(
    image: np.ndarray,
    n_scales: int = 3,
    n_orientations: int = 6,
    noise_threshold: float = None,
    noise_std_factor: float = 3.0,
    epsilon: float = 1e-5,
    min_wavelength: float = 6.0,
    mult: float = 2.1,
    sigma_on_f: float = 0.55,
    d_theta_on_sigma: float = 1.5,
) -> tuple:
    img = image.astype(np.float64)
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())

    H, W = img.shape
    fb   = log_gabor_filter_bank(
        (H, W), n_scales, n_orientations, min_wavelength, mult, sigma_on_f, d_theta_on_sigma
    )
    even_resp, odd_resp = compute_even_odd_responses(img, fb)

    pc_per_o  = np.zeros((n_orientations, H, W), dtype=np.float64)
    amp_per_o = np.zeros((n_orientations, H, W), dtype=np.float64)

    for o in range(n_orientations):
        F     = even_resp[:, o, :, :].sum(axis=0)
        H_vec = odd_resp [:, o, :, :].sum(axis=0)
        E_total = np.sqrt(F ** 2 + H_vec ** 2) + epsilon
        phi_e = F     / E_total
        phi_o = H_vec / E_total

        T = noise_threshold
        num_o = np.zeros((H, W))
        den_o = np.zeros((H, W))

        for s in range(n_scales):
            e_so = even_resp[s, o, :, :]
            o_so = odd_resp [s, o, :, :]
            A_so = np.sqrt(e_so ** 2 + o_so ** 2)

            dot   = e_so * phi_e + o_so * phi_o
            cross = np.abs(e_so * phi_o - o_so * phi_e)
            sharp = dot - cross

            if T is None and s == 0:
                noise_power = np.median(np.abs(A_so)) / 0.6745
                T = noise_std_factor * noise_power

            T_use = T if T is not None else 0.0
            W_o   = 1.0 / (1.0 + np.exp(3.0 * (0.35 - A_so / (E_total + epsilon))))
            num_o += np.maximum(W_o * sharp - T_use, 0.0)
            den_o += A_so
            amp_per_o[o] += A_so

        pc_per_o[o] = num_o / (den_o + epsilon)

    return pc_per_o, amp_per_o


def moment_analysis(pc_per_orientation: np.ndarray) -> tuple:
    n_orientations = pc_per_orientation.shape[0]
    thetas = np.array([o * np.pi / n_orientations for o in range(n_orientations)])
    PC_cos = pc_per_orientation * np.cos(thetas)[:, None, None]
    PC_sin = pc_per_orientation * np.sin(thetas)[:, None, None]
    a = (PC_cos ** 2).sum(axis=0)
    b = 2.0 * (PC_cos * PC_sin).sum(axis=0)
    c = (PC_sin ** 2).sum(axis=0)
    disc = np.sqrt(np.maximum(b ** 2 + (a - c) ** 2, 0.0))
    return 0.5 * (c + a + disc), 0.5 * (c + a - disc)


def compute_mim(amplitude_per_orientation: np.ndarray) -> np.ndarray:
    return np.argmax(amplitude_per_orientation, axis=0).astype(np.uint8)


def mim_descriptor(
    mim: np.ndarray,
    keypoints: np.ndarray,
    grid_size: int = 6,
    num_orientation_bins: int = 6,
    patch_radius: int = 24,
) -> np.ndarray:
    N   = len(keypoints)
    dim = grid_size * grid_size * num_orientation_bins
    descriptors = np.zeros((N, dim), dtype=np.float32)
    H, W = mim.shape
    p    = patch_radius
    pw   = 2 * p
    gy, gx = np.meshgrid(np.arange(pw), np.arange(pw), indexing='ij')
    gauss  = np.exp(-((gx - p) ** 2 + (gy - p) ** 2) / (2.0 * p ** 2))
    cell_h = pw // grid_size
    cell_w = pw // grid_size

    for n, (kx, ky) in enumerate(keypoints):
        kx, ky = int(round(kx)), int(round(ky))
        if kx < p or kx >= W - p or ky < p or ky >= H - p:
            continue
        patch = mim[ky - p : ky + p, kx - p : kx + p].astype(np.float32) * gauss
        desc_idx = 0
        for ci in range(grid_size):
            for cj in range(grid_size):
                cell = patch[ci*cell_h:(ci+1)*cell_h, cj*cell_w:(cj+1)*cell_w].ravel()
                hist, _ = np.histogram(cell, bins=num_orientation_bins, range=(0, num_orientation_bins))
                descriptors[n, desc_idx:desc_idx+num_orientation_bins] = hist
                desc_idx += num_orientation_bins
        norm = np.linalg.norm(descriptors[n]) + 1e-8
        descriptors[n] /= norm

    return descriptors


def extract_structural_features(image: np.ndarray, n_scales: int = 3, n_orientations: int = 6) -> dict:
    pc_per_o, amp_per_o = _phase_congruency_per_orientation(
        image, n_scales=n_scales, n_orientations=n_orientations
    )
    pc_map = np.clip(pc_per_o.sum(axis=0) / n_orientations, 0, 1)
    edge_map, corner_map = moment_analysis(pc_per_o)
    mim = compute_mim(amp_per_o)
    return {'pc_map': pc_map, 'mim': mim, 'edge_map': edge_map, 'corner_map': corner_map, 'amp_per_o': amp_per_o}
