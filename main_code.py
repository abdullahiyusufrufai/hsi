import time
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

# OpenHSI imports
from openhsi.capture import CameraProperties, ProcessDatacube
from openhsi.data import load_calibration
from openhsi.io import save_calibration, save_datacube

# Optional hardware
PICAM_AVAILABLE = False
try:
    from picamera2 import Picamera2
    PICAM_AVAILABLE = True
except ImportError:
    pass

# Optional preview
MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass


class OpenHSITimedScanner:
    def __init__(
        self,
        config_dir: str = "openhsi_config",
        target_resolution: Tuple[int, int] = (224, 224),
        scan_positions: int = 10,
        interval_sec: float = 1.0,
    ):
        """
        Fully automatic timed hyperspectral scanner.
        Captures a frame every `interval_sec` seconds for `scan_positions` steps.
        No manual input. No Arduino. Pure software timing.
        """
        self.target_resolution = target_resolution  # (H, W)
        self.scan_positions = scan_positions
        self.interval_sec = interval_sec
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

        self.cam_props = None
        self.calibration = None
        self.scan_images = []
        self.final_cube = None

        self._setup_logging()
        self.logger.info("Timed HSI Scanner initialized")

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config_dir / "timed_scan.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def initialize_openhsi(self):
        settings_path = self.config_dir / "settings.json"
        cal_path = self.config_dir / "calibration.nc"

        if not settings_path.exists():
            self._create_default_settings(settings_path)
        if not cal_path.exists():
            self.logger.warning("No calibration found. Creating minimal NetCDF calibration.")
            self._create_minimal_calibration(cal_path)

        try:
            self.cam_props = CameraProperties(
                json_path=str(settings_path),
                cal_path=str(cal_path)
            )
            self.calibration = load_calibration(str(cal_path))
            self.logger.info("OpenHSI initialized")
            return True
        except Exception as e:
            self.logger.error(f"OpenHSI init failed: {e}")
            return False

    def _create_default_settings(self, path: Path):
        H, W = self.target_resolution
        settings = {
            "camera": {
                "name": "TimedScanner",
                "sensor_type": "snapshot",
                "spatial_pixels": W,
                "spectral_pixels": H,
                "bit_depth": 12,
            },
            "processing": {
                "default_processing_level": 4,
                "output_units": "radiance"
            }
        }
        with open(path, 'w') as f:
            json.dump(settings, f, indent=4)

    def _create_minimal_calibration(self, path: Path):
        from openhsi.io import save_calibration
        H, W = self.target_resolution
        wavelengths = np.linspace(400, 900, self.scan_positions).astype(np.float32)
        cal_data = {
            "wavelengths": wavelengths,
            "dark_current": np.zeros((H, W), dtype=np.float32),
            "flat_field": np.ones((H, W), dtype=np.float32),
            "rad_fit": np.ones(self.scan_positions, dtype=np.float32) * 100.0,
            "calibration_date": datetime.now().isoformat(),
            "notes": "Minimal timed-scan calibration"
        }
        save_calibration(str(path), cal_data)

    def _capture_frame(self) -> np.ndarray:
        """Capture or simulate a single frame."""
        if PICAM_AVAILABLE:
            picam = Picamera2()
            config = picam.create_still_configuration(
                main={"size": (self.target_resolution[1], self.target_resolution[0]), "format": "RGB888"}
            )
            picam.configure(config)
            picam.start()
            time.sleep(0.2)  # stabilize
            frame = picam.capture_array()
            picam.stop()
            if frame.ndim == 3:
                gray = np.mean(frame, axis=2)
            else:
                gray = frame
            return gray.astype(np.float32)
        else:
            # Simulated frame
            H, W = self.target_resolution
            t = time.time()
            x = np.linspace(0, 4 * np.pi, W)
            y = np.sin(x + t * 0.5)  # slowly varying pattern
            noise = np.random.normal(0, 5, (H, W))
            return (np.tile(y, (H, 1)) * 50 + 100 + noise).astype(np.float32)

    def run_timed_scan(self):
        """Automatically capture frames at fixed intervals."""
        self.scan_images = []
        self.logger.info(f"Starting timed scan: {self.scan_positions} frames, interval={self.interval_sec}s")

        for i in range(self.scan_positions):
            self.logger.info(f"Capturing frame {i+1}/{self.scan_positions}")
            frame = self._capture_frame()
            self.scan_images.append(frame)

            if i < self.scan_positions - 1:  # don't sleep after last frame
                time.sleep(self.interval_sec)

        self.logger.info("Timed scan completed.")

    def process_and_save(self, sample_name: Optional[str] = None):
        if not self.scan_images:
            self.logger.error("No images to process")
            return False

        # Assemble cube: (H, W, N_bands)
        cube = np.stack(self.scan_images, axis=2).astype(np.float32)
        self.logger.info(f"Cube shape: {cube.shape}")

        # Process with OpenHSI
        try:
            processor = ProcessDatacube(
                data=cube,
                camera_props=self.cam_props,
                processing_lvl=4  # radiance
            )
            self.final_cube = processor.process()
        except Exception as e:
            self.logger.warning(f"OpenHSI processing failed: {e}. Using raw cube.")
            self.final_cube = cube

        # Save
        if sample_name is None:
            sample_name = datetime.now().strftime("timed_scan_%Y%m%d_%H%M%S")
        output_dir = Path("results") / sample_name
        output_dir.mkdir(parents=True, exist_ok=True)

        wavelengths = self.calibration.get("wavelengths") if self.calibration else None
        save_datacube(
            self.final_cube,
            output_dir / f"{sample_name}.nc",
            metadata={"sample_name": sample_name, "acq_mode": "timed_scan"},
            wavelengths=wavelengths
        )

        self._save_rgb_preview(output_dir, sample_name)
        self.logger.info(f"Results saved to: {output_dir}")
        return True

    def _save_rgb_preview(self, output_dir: Path, name: str):
        if not MATPLOTLIB_AVAILABLE or self.final_cube is None:
            return
        try:
            cube = self.final_cube
            n_bands = cube.shape[2]
            r = cube[:, :, -1]
            g = cube[:, :, n_bands // 2]
            b = cube[:, :, 0]
            rgb = np.stack([r, g, b], axis=2)
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
            plt.imsave(output_dir / "rgb_preview.png", np.clip(rgb, 0, 1))
        except Exception as e:
            self.logger.warning(f"RGB preview failed: {e}")

    def run_full_acquisition(self, sample_name: Optional[str] = None):
        if not self.initialize_openhsi():
            return False
        self.run_timed_scan()
        return self.process_and_save(sample_name)


# ======================
# Main
# ======================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="OpenHSI Timed Scanner")
    parser.add_argument("--positions", type=int, default=10, help="Number of frames to capture")
    parser.add_argument("--interval", type=float, default=1.0, help="Interval between frames (seconds)")
    parser.add_argument("--name", type=str, default=None, help="Sample name")
    parser.add_argument("--resolution", type=str, default="224,224", help="H,W resolution")
    args = parser.parse_args()

    H, W = map(int, args.resolution.split(","))
    scanner = OpenHSITimedScanner(
        scan_positions=args.positions,
        interval_sec=args.interval,
        target_resolution=(H, W)
    )

    print(f"\n🚀 Starting timed acquisition:")
    print(f"   Frames: {args.positions}")
    print(f"   Interval: {args.interval} s")
    print(f"   Resolution: {H}x{W}")
    print(f"   Sample name: {args.name or 'auto'}\n")

    success = scanner.run_full_acquisition(args.name)
    if success:
        print("\n✅ Done! Results saved.")
    else:
        print("\n❌ Failed.")


if __name__ == "__main__":
    main()
