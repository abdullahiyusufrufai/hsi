## main HSI code 

import os
import time
import numpy as np
import json
import h5py
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Tuple

# OpenHSI imports
from openhsi.capture import CameraProperties, ProcessDatacube
from openhsi.data import load_calibration, save_calibration
import openhsi.utils as utils

# Hardware imports
import RPi.GPIO as GPIO
try:
    from picamera2 import Picamera2
    PICAM_AVAILABLE = True
except ImportError:
    PICAM_AVAILABLE = False
    print("Warning: picamera2 not available")

# GPIO Configuration
GPIO_TRIGGER = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)

class OpenHSIPushPullSystem:
    def __init__(self, config_dir: str = "openhsi_config", target_resolution: Tuple[int, int] = (224, 224)):
        """
        Push-pull HSI system using OpenHSI methods and calibration.
        
        Args:
            config_dir: Directory containing OpenHSI config files
            target_resolution: (spectral_pixels, spatial_pixels)
        """
        self.target_resolution = target_resolution
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # OpenHSI camera properties
        self.cam_props = None
        self.calibration = None
        self.picam = None
        
        # Scanning parameters
        self.scan_positions = 50   # Total number of spatial lines to capture
        self.movement_delay = 0.5  # Fixed time (seconds) to wait for motor move
        self.processing_lvl = 4    # Radiance output level
        
        # Image buffers
        self.scan_images = []
        self.final_cube = None
        self.scanning = False
        
        self.setup_logging()
        
        print("\n" + "="*60)
        print("OPENHSI PUSH-PULL HYPERSPECTRAL IMAGING SYSTEM")
        print("="*60)
        print(f"Config: {self.config_dir} | Target Res: {target_resolution}")
        
    def setup_logging(self):
        """Configure logging for the system."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config_dir / 'pushpull_hsi.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def initialize_open_hsi(self):
        """Initialize OpenHSI camera properties and load/create calibration."""
        try:
            settings_path = self.config_dir / "settings.json"
            cal_path = self.config_dir / "calibration.pkl"
            
            if not settings_path.exists():
                self.create_default_settings(settings_path)
            
            if not cal_path.exists():
                self.create_minimal_calibration(cal_path)
            
            self.cam_props = CameraProperties(json_path=str(settings_path), cal_path=str(cal_path))
            self.calibration = load_calibration(str(cal_path))
            self.logger.info("OpenHSI system initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenHSI: {e}")
            return False

    def create_default_settings(self, settings_path: Path):
        """Create a default OpenHSI settings JSON."""
        settings = {
            "camera": {
                "name": "PushPullScanner",
                "sensor_type": "pushbroom",
                "spatial_pixels": self.target_resolution[1],
                "spectral_pixels": self.target_resolution[0],
                "bit_depth": 12,
                "exposure_mode": "manual",
                "gain": 1.0
            },
            "scanning": {
                "scan_direction": "vertical",
                "frame_rate": 10,
                "trigger_mode": "external"
            },
            "processing": {
                "default_processing_level": self.processing_lvl,
                "binning_mode": "fast",
                "output_units": "radiance"
            },
            "calibration": {
                "dark_correction": True,
                "flat_field_correction": True,
                "wavelength_calibration": True
            }
        }
        with open(settings_path, 'w') as f:
            json.dump(settings, f, indent=4)

    def create_minimal_calibration(self, cal_path: Path):
        """Create a placeholder calibration pickle file."""
        cal_data = {
            "wavelengths": np.linspace(400, 800, self.target_resolution[0]).astype(np.float32),
            "dark_current": np.zeros((self.target_resolution[0], self.target_resolution[1]), dtype=np.float32),
            "flat_field": np.ones((self.target_resolution[0], self.target_resolution[1]), dtype=np.float32),
            "rad_fit": np.ones((self.target_resolution[0],), dtype=np.float32) * 100.0,
            "smile_correction": np.zeros((self.target_resolution[0], self.target_resolution[1]), dtype=np.float32),
            "calibration_date": datetime.now().isoformat(),
            "calibration_notes": "PLACEHOLDER - PLEASE CALIBRATE"
        }
        save_calibration(str(cal_path), cal_data)

    def capture_pushpull_scan(self) -> bool:
        """Pulse Arduino and capture frames based on a fixed time delay."""
        if not PICAM_AVAILABLE:
            self.logger.error("Picamera2 not found.")
            return False
        
        self.logger.info("Starting scan sequence (Fixed Delay Mode)...")
        self.scanning = True
        self.scan_images = []
        
        try:
            self.picam = Picamera2()
            config = self.picam.create_still_configuration(
                main={"size": (self.target_resolution[1], self.target_resolution[0]), "format": "RGB888"}
            )
            self.picam.configure(config)
            self.picam.start()
            
            # Pulse the Arduino once to start its internal stepper routine
            GPIO.output(GPIO_TRIGGER, GPIO.HIGH)
            time.sleep(0.1)
            GPIO.output(GPIO_TRIGGER, GPIO.LOW)
            
            for position in range(self.scan_positions):
                if not self.scanning: break
                
                # Capture frame and convert to grayscale (raw intensity)
                frame = self.picam.capture_array()
                if frame.ndim == 3:
                    gray = np.mean(frame, axis=2).astype(np.float32)
                else:
                    gray = frame.astype(np.float32)
                
                self.scan_images.append(gray)
                self.logger.info(f"Captured line {position+1}/{self.scan_positions}")
                
                # Wait for motor to step to next position
                time.sleep(self.movement_delay)
                
            return True
        except Exception as e:
            self.logger.error(f"Scanning Error: {e}")
            return False
        finally:
            self.cleanup_camera()

    def stitch_and_process_openhsi(self):
        """Reconstruct 3D cube and apply OpenHSI processing."""
        if not self.scan_images:
            self.logger.error("No image data captured.")
            return None
        
        try:
            # Reconstruct cube: (Spatial_Y, Spatial_X, Spectral_Lambda)
            # This stacks our captured lines along the Y axis
            cube = np.stack(self.scan_images, axis=0)
            
            processor = ProcessDatacube(
                data=cube,
                camera_props=self.cam_props,
                processing_lvl=self.processing_lvl
            )
            
            self.final_cube = processor.process()
            self.logger.info(f"Processing complete. Cube shape: {self.final_cube.shape}")
            return self.final_cube
        except Exception as e:
            self.logger.error(f"OpenHSI Processing Error: {e}")
            return None

    def save_results(self, sample_name: str = None):
        """Save the processed datacube in OpenHSI NetCDF format."""
        if self.final_cube is None: 
            return None
        
        sample_name = sample_name or f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = Path("results") / sample_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            from openhsi.io import save_datacube
            file_path = output_dir / f"{sample_name}.nc"
            save_datacube(
                self.final_cube, 
                file_path,
                wavelengths=self.calibration.get('wavelengths', None)
            )
            self.logger.info(f"Results saved to {file_path}")
            return str(output_dir)
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            return None

    def cleanup_camera(self):
        """Safely release the camera resource."""
        if self.picam:
            try:
                self.picam.stop()
                self.picam.close()
            except: 
                pass
            self.picam = None

    def cleanup(self):
        """Clean up all hardware resources."""
        self.cleanup_camera()
        GPIO.cleanup()
        self.logger.info("Hardware cleanup complete.")

    def run_full_pipeline(self, sample_name: str = "Test_Scan"):
        """Executes initialization, capture, processing, and saving."""
        if self.initialize_open_hsi():
            if self.capture_pushpull_scan():
                if self.stitch_and_process_openhsi():
                    return self.save_results(sample_name)
        return False

# Main Execution
if __name__ == "__main__":
    scanner = OpenHSIPushPullSystem()
    try:
        scanner.run_full_pipeline(sample_name="PushPull_Session_01")
    except KeyboardInterrupt:
        print("\nScan interrupted by user.")
    finally:
        scanner.cleanup()
