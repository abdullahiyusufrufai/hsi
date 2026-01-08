import numpy as np
from typing import Optional

def apply_dark_white_correction(
    cube: np.ndarray,
    dark: np.ndarray,
    white: np.ndarray
) -> np.ndarray:
    """Apply basic reflectance correction."""
    dark_3d = dark[:, np.newaxis, :]
    white_3d = white[:, np.newaxis, :]
    corrected = (cube - dark_3d) / (white_3d - dark_3d + 1e-6)
    return np.clip(corrected, 0, 1.5)
