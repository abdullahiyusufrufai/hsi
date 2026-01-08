import numpy as np
from pathlib import Path
from typing import Optional
from openhsi_pushpull.calibration.references import load_calibration
from openhsi_pushpull.processing.corrections import apply_dark_white_correction

class DataProcessor:
    def __init__(self, cal_file: Optional[Path] = None):
        self.cal_file = cal_file
        self.cal_data = None
        if cal_file:
            self.cal_data = load_calibration(cal_file)

    def build_cube(self, scan_lines: list, spectral_bands: int) -> np.ndarray:
        """Assemble lines into (H, W, B) cube."""
        spatial_stack = np.stack(scan_lines, axis=1)  # (H, W)
        # Replicate spatial line across spectral bands (placeholder)
        cube = np.repeat(spatial_stack[:, :, np.newaxis], spectral_bands, axis=2)
        return cube.astype(np.float32)

    def process(self, cube: np.ndarray) -> np.ndarray:
        """Apply calibration if available."""
        if self.cal_data is not None:
            return apply_dark_white_correction(
                cube, self.cal_data["dark"], self.cal_data["white"]
            )
        else:
            return cube
