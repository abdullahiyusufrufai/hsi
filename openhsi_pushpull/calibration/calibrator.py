import logging
from pathlib import Path
from typing import Optional
import numpy as np
from openhsi_pushpull.acquisition.scanner import PushBroomScanner
from openhsi_pushpull.calibration.references import save_calibration

logger = logging.getLogger(__name__)


class Calibrator:
    def __init__(self, scanner: PushBroomScanner, cal_dir: str = "calibration"):
        self.scanner = scanner
        self.cal_dir = Path(cal_dir)

    def capture_reference(self, name: str, frames: int = 20) -> Optional[np.ndarray]:
        """Capture and median-combine frames for a reference target."""
        lines = []
        logger.info(f"Capturing {name} reference ({frames} frames)")
        for _ in range(frames):
            # Simulate: each "frame" is actually same line (since we can't move during cal)
            line = self.scanner.capture_lines(1)
            if line:
                lines.append(line[0])
            else:
                return None
        return np.median(lines, axis=0) if lines else None

    def run_calibration(self) -> Optional[Path]:
        """Run full dark/white calibration."""
        # Dark reference
        logger.info("=== DARK REFERENCE ===")
        input("Cover lens. Press Enter to start dark capture...")
        dark = self.capture_reference("dark")
        if dark is None:
            return None

        # White reference
        logger.info("=== WHITE REFERENCE ===")
        input("Place white reference. Press Enter...")
        white = self.capture_reference("white")
        if white is None:
            return None

        # Wavelengths (placeholder)
        wavelengths = np.linspace(400, 1000, self.scanner.spectral_bands, dtype=np.float32)

        # Save
        cal_file = save_calibration(self.cal_dir, dark, white, wavelengths)
        logger.info(f"Calibration saved to {cal_file}")
        return cal_file
