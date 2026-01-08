"""

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
        
        
"""



import time
import logging
from typing import List, Optional
import numpy as np

try:
    from picamera2 import Picamera2
    PICAM_AVAILABLE = True
except ImportError:
    PICAM_AVAILABLE = False

logger = logging.getLogger(__name__)


class PushBroomScanner:
    """
    Pi Camera scanner for pushbroom hyperspectral imaging.
    
    Assumes:
    - Camera captures full frames.
    - A single column (e.g., column 0) is used as the spatial line.
    - Motion stage moves perpendicular to this line.
    """

    def __init__(
        self,
        spatial_height: int = 480,
        spectral_bands: int = 256,
        exposure_time_us: int = 100_000,
        step_delay_s: float = 0.5,
    ):
        # Validate spatial height (IMX219 max in 640x mode is 480)
        if spatial_height > 480:
            logger.warning(f"Spatial height {spatial_height} > 480; clamping to 480")
            spatial_height = 480
        if spatial_height <= 0:
            raise ValueError("spatial_height must be positive")

        self.spatial_height = spatial_height
        self.spectral_bands = spectral_bands
        self.exposure_time_us = exposure_time_us
        self.step_delay_s = step_delay_s
        self._picam: Optional[Picamera2] = None

    def initialize(self) -> bool:
        """Initialize camera with valid configuration for IMX219."""
        if not PICAM_AVAILABLE:
            logger.error("picamera2 not available")
            return False

        try:
            self._picam = Picamera2()
            # Use smallest full-res mode: 640x480
            # Width must be >= some minimum (640 is safe)
            main_config = {
                "size": (640, self.spatial_height),  # (width, height)
                "format": "RGB888"
            }
            config = self._picam.create_still_configuration(main=main_config)
            self._picam.configure(config)
            self._picam.set_controls({
                "ExposureTime": self.exposure_time_us,
                "AnalogueGain": 1.0,
                "AeEnable": False,   # Manual exposure
                "AwbEnable": False   # Manual white balance
            })
            self._picam.start()
            time.sleep(1.0)  # Allow sensor to stabilize
            logger.info(f"Camera initialized: {main_config}")
            return True
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            self._safe_close()
            return False

    def _safe_close(self):
        """Safely close the camera if it exists."""
        if self._picam is not None:
            try:
                self._picam.stop()
                self._picam.close()
            except Exception as e:
                logger.warning(f"Error closing camera: {e}")
            finally:
                self._picam = None

    def capture_lines(self, num_lines: int) -> List[np.ndarray]:
        """
        Capture num_lines spatial lines.
        Each line is extracted as column 0 of a grayscale frame.
        """
        if not self.initialize():
            logger.error("Failed to initialize camera")
            return []

        lines: List[np.ndarray] = []
        logger.info(f"Starting capture of {num_lines} lines")

        try:
            for i in range(num_lines):
                # Capture full frame (H, W, 3)
                raw_frame = self._picam.capture_array("main")
                logger.debug(f"Captured frame shape: {raw_frame.shape}")

                # Convert to grayscale: (H, W, 3) → (H, W)
                if raw_frame.ndim == 3:
                    gray_frame = np.mean(raw_frame, axis=2).astype(np.float32)
                else:
                    gray_frame = raw_frame.astype(np.float32)

                # Extract first column as spatial line: (H, W) → (H,)
                if gray_frame.shape[1] > 0:
                    line = gray_frame[:, 0]
                else:
                    logger.error("Captured frame has zero width")
                    break

                lines.append(line)
                logger.debug(f"Line {i+1}/{num_lines} captured: mean={line.mean():.1f}")

                # Delay to sync with motion stage (except after last line)
                if i < num_lines - 1 and self.step_delay_s > 0:
                    time.sleep(self.step_delay_s)

        except Exception as e:
            logger.error(f"Error during capture: {e}")
        finally:
            self._safe_close()

        logger.info(f"Capture complete: {len(lines)} lines acquired")
        return lines
