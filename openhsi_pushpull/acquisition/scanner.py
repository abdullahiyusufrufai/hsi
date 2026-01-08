import time
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

try:
    from picamera2 import Picamera2
    PICAM_AVAILABLE = True
except ImportError:
    PICAM_AVAILABLE = False

logger = logging.getLogger(__name__)


class PushBroomScanner:
    def __init__(
        self,
        spatial_height: int = 512,
        spectral_bands: int = 256,
        exposure_time_us: int = 100_000,
        step_delay_s: float = 0.5,
    ):
        self.spatial_height = spatial_height
        self.spectral_bands = spectral_bands
        self.exposure_time_us = exposure_time_us
        self.step_delay_s = step_delay_s
        self._picam = None

    def initialize(self) -> bool:
        if not PICAM_AVAILABLE:
            logger.error("picamera2 not available")
            return False

        try:
            self._picam = Picamera2()
            # Configure for 1-pixel-wide ROI (pushbroom line)
            main_config = {"size": (1, self.spatial_height), "format": "SBGGR10"}
            config = self._picam.create_still_configuration(main=main_config)
            self._picam.configure(config)
            self._picam.set_controls({"ExposureTime": self.exposure_time_us, "AnalogueGain": 1.0})
            self._picam.start()
            time.sleep(1.0)
            logger.info("Camera initialized for pushbroom scanning")
            return True
        except Exception as e:
            logger.error(f"Camera init failed: {e}")
            self._safe_close()
            return False

    def _safe_close(self):
        if self._picam:
            try:
                self._picam.stop()
                self._picam.close()
            except Exception as e:
                logger.warning(f"Error closing camera: {e}")
            finally:
                self._picam = None

    def capture_lines(self, num_lines: int) -> List[np.ndarray]:
        """Capture num_lines lines with delay between each."""
        if not self.initialize():
            return []

        lines = []
        logger.info(f"Starting capture of {num_lines} lines")

        try:
            for i in range(num_lines):
                raw = self._picam.capture_array("main")
                # Extract 1D spatial line
                if raw.ndim == 3:
                    line = raw[:, 0, 0]
                elif raw.ndim == 2:
                    line = raw[:, 0]
                else:
                    line = raw
                lines.append(line.astype(np.float32))
                logger.debug(f"Captured line {i+1}/{num_lines}")

                if i < num_lines - 1 and self.step_delay_s > 0:
                    time.sleep(self.step_delay_s)
        except Exception as e:
            logger.error(f"Capture failed: {e}")
        finally:
            self._safe_close()

        return lines
