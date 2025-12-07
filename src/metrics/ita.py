import numpy as np

def compute_ita_from_lab(l_star: float, b_star: float) -> float:
    """
    Compute Individual Typology Angle (ITA) from a single CIELAB (L*, b*).
    Formula:
        ITA = (180 / pi) * arctan((L* - 50) / b*)
    """
    # avoid division by zero
    b_star = np.clip(b_star, 1e-6, None)
    ita = (180.0 / np.pi) * np.arctan((l_star - 50.0) / b_star)
    return float(ita)

def compute_ita_from_patch(lab_patch: np.ndarray) -> float:
    """
    Compute ITA from a LAB image patch of shape (H, W, 3)
    by averaging L* and b* over the patch.
    """
    if lab_patch.ndim != 3 or lab_patch.shape[2] != 3:
        raise ValueError("lab_patch must have shape (H, W, 3)")
    l_channel = lab_patch[:, :, 0]
    b_channel = lab_patch[:, :, 2]
    l_mean = float(np.mean(l_channel))
    b_mean = float(np.mean(b_channel))
    return compute_ita_from_lab(l_mean, b_mean)
