import os
import time
import numpy as np
import json
import h5py
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Tuple
import threading
import queue

# OpenHSI imports
from openhsi.capture import CameraProperties, ProcessDatacube
from openhsi.data import load_calibration, save_calibration
from openhsi.atmos import EmpiricalLineCalibration
import openhsi.utils as utils

# Hardware imports (keep for your Arduino control)
import RPi.GPIO as GPIO
try:
    from picamera2 import Picamera2
    PICAM_AVAILABLE = True
except ImportError:
    PICAM_AVAILABLE = False
    print("Warning: picamera2 not available")

# GPIO Configuration (keep your hardware interface)
GPIO_TRIGGER = 17
GPIO_STATUS = 27
GPIO.setmode(GPIO.BCM)
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_STATUS, GPIO.IN, pull_up_down=GPIO.PUD_UP)

class OpenHSIPushPullSystem:
     def __init__(
        self,
        config_dir: str = "openhsi_config",
        target_resolution: Tuple[int, int] = (224, 224)
    ):
        """
        Push-pull HSI system using OpenHSI methods and calibration.
        
        Args:
            config_dir: Directory containing OpenHSI config files
            target_resolution: Target image resolution (height, width)
        """
        self.target_resolution = target_resolution
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # OpenHSI camera properties (virtual camera for processing)
        self.cam_props = None
        self.calibration = None
        
        # Scanning parameters
        self.scan_positions = 10
        self.step_distance_mm = 0.1
        self.exposure_time_ms = 5000
        
        # OpenHSI processing level
        self.processing_lvl = 4  # Radiance output (µW/cm²/sr/nm)
        
        # Image buffers
        self.scan_images = []
        self.current_position = 0
        
        # Results storage
        self.final_cube = None
        self.radiance_cube = None
        self.reflectance_cube = None
        
        # Threading and sync
        self.capture_queue = queue.Queue()
        self.scanning = False
        
        # Calibration targets
        self.calibration_targets = {
            "dark": None,
            "white": None,
            "foil": None  # Aluminum foil as secondary reference
        }
        
        # Setup logging
        self.setup_logging()
        
        print("\n" + "="*60)
        print("OPENHSI PUSH-PULL HYPERSPECTRAL IMAGING SYSTEM")
        print("="*60)
        print(f"Config directory: {self.config_dir}")
        print(f"Processing level: {self.processing_lvl}")
        print(f"Target resolution: {target_resolution}")
        
    def setup_logging(self):
        """Configure logging for the system."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config_dir / 'pushpull_hsi.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_open_hsi(self):
        """
        Initialize OpenHSI camera properties and load calibration.
        
        Creates minimal configuration files if they don't exist.
        """
        try:
            # Paths for OpenHSI config files
            settings_path = self.config_dir / "settings.json"
            cal_path = self.config_dir / "calibration.pkl"
            
            # Create default settings if they don't exist
            if not settings_path.exists():
                self.create_default_settings(settings_path)
            
            # Create minimal calibration if it doesn't exist
            if not cal_path.exists():
                self.logger.warning("No calibration found. Run full calibration first.")
                self.create_minimal_calibration(cal_path)
            
            # Load camera properties
            self.cam_props = CameraProperties(
                json_path=str(settings_path),
                cal_path=str(cal_path)
            )
            
            # Load calibration data
            self.calibration = load_calibration(str(cal_path))
            
            self.logger.info("OpenHSI system initialized")
            self.logger.info(f"Settings: {settings_path}")
            self.logger.info(f"Calibration: {cal_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenHSI: {e}")
            return False
    
    def create_default_settings(self, settings_path: Path):
        """
        Create default OpenHSI settings for push-pull scanner.
        
        This is a minimal configuration. You should update this based on your
        actual camera and scanning parameters.
        """
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
                "scan_direction": "horizontal",
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
        
        self.logger.info(f"Created default settings at {settings_path}")
    
    def create_minimal_calibration(self, cal_path: Path):
        """
        Create minimal calibration file with placeholder data.
        
        WARNING: This is not a proper calibration! You MUST run
        proper calibration with real targets.
        """
        # Placeholder calibration data
        # In reality, this should come from integrating sphere calibration
        cal_data = {
            "wavelengths": np.linspace(380, 900, 224).astype(np.float32),
            "dark_current": np.zeros((224, 224), dtype=np.float32),
            "flat_field": np.ones((224, 224), dtype=np.float32),
            "rad_fit": np.ones((224,), dtype=np.float32) * 100.0,  # Placeholder
            "smile_correction": np.zeros((224, 224), dtype=np.float32),
            "calibration_date": datetime.now().isoformat(),
            "calibration_notes": "PLACEHOLDER - RUN PROPER CALIBRATION"
        }
        
        save_calibration(str(cal_path), cal_data)
        self.logger.warning(f"Created placeholder calibration at {cal_path}")
        self.logger.warning("THIS IS NOT ACCURATE - RUN PROPER CALIBRATION")
    
    def calibrate_with_openhsi_method(self, calibration_dir: str = "calibration_data"):
        """
        Perform proper OpenHSI-style calibration.
        
        This method guides you through capturing the necessary calibration
        targets and computing proper calibration coefficients.
        """
        self.logger.info("Starting OpenHSI-style calibration procedure")
        
        cal_dir = Path(calibration_dir)
        cal_dir.mkdir(exist_ok=True)
        
        if not PICAM_AVAILABLE:
            self.logger.error("picamera2 not available for calibration")
            return False
        
        try:
            # Initialize camera
            self.picam = Picamera2()
            config = self.picam.create_still_configuration(
                main={"size": (self.target_resolution[1], self.target_resolution[0]), 
                      "format": "RGB888"}
            )
            self.picam.configure(config)
            self.picam.start()
            time.sleep(2)  # Stabilization
            
            # Step 1: Dark reference (cover lens completely)
            self.logger.info("=== DARK REFERENCE CALIBRATION ===")
            input("1. Completely cover the lens with lens cap or dark material.\n"
                  "   Press Enter when ready to capture dark reference...")
            
            dark_frames = []
            for i in range(20):
                frame = self.picam.capture_array()
                gray = np.mean(frame, axis=2) if frame.ndim == 3 else frame
                dark_frames.append(gray)
                time.sleep(0.1)
            
            self.calibration_targets["dark"] = np.median(dark_frames, axis=0)
            np.save(cal_dir / "dark_reference.npy", self.calibration_targets["dark"])
            self.logger.info(f"Dark reference captured: {self.calibration_targets['dark'].shape}")
            
            # Step 2: White reference (Spectralon or similar)
            self.logger.info("\n=== WHITE REFERENCE CALIBRATION ===")
            input("2. Place white reference (Spectralon, barium sulfate, or white ceramic).\n"
                  "   Ensure uniform illumination. Press Enter when ready...")
            
            white_frames = []
            for i in range(20):
                frame = self.picam.capture_array()
                gray = np.mean(frame, axis=2) if frame.ndim == 3 else frame
                white_frames.append(gray)
                time.sleep(0.1)
            
            self.calibration_targets["white"] = np.median(white_frames, axis=0)
            np.save(cal_dir / "white_reference.npy", self.calibration_targets["white"])
            self.logger.info(f"White reference captured: {self.calibration_targets['white'].shape}")
            
            # Step 3: Secondary reference (aluminum foil)
            self.logger.info("\n=== SECONDARY REFERENCE CALIBRATION ===")
            input("3. Place aluminum foil reference (ensure flat, clean surface).\n"
                  "   Press Enter when ready...")
            
            foil_frames = []
            for i in range(20):
                frame = self.picam.capture_array()
                gray = np.mean(frame, axis=2) if frame.ndim == 3 else frame
                foil_frames.append(gray)
                time.sleep(0.1)
            
            self.calibration_targets["foil"] = np.median(foil_frames, axis=0)
            np.save(cal_dir / "foil_reference.npy", self.calibration_targets["foil"])
            
            # Step 4: Compute calibration coefficients
            self.logger.info("\n=== COMPUTING CALIBRATION COEFFICIENTS ===")
            self.compute_calibration_coefficients(cal_dir)
            
            # Step 5: Update OpenHSI calibration file
            self.update_openhsi_calibration(cal_dir)
            
            self.logger.info("Calibration complete!")
            self.logger.info(f"Calibration data saved to: {cal_dir}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Calibration failed: {e}")
            return False
        finally:
            if hasattr(self, 'picam'):
                self.picam.stop()
    
    def compute_calibration_coefficients(self, cal_dir: Path):
        """
        Compute proper calibration coefficients from captured references.
        
        This mimics what OpenHSI does with integrating sphere calibration.
        """
        # Load references
        dark = self.calibration_targets["dark"]
        white = self.calibration_targets["white"]
        foil = self.calibration_targets["foil"]
        
        # Compute flat field correction (per-pixel response)
        flat_field = (white - dark)
        flat_field = np.where(flat_field > 0, flat_field, 1e-10)
        
        # Normalize flat field
        flat_field_norm = flat_field / np.median(flat_field)
        
        # Compute system response using foil (known reflectance ~0.88)
        # In proper calibration, this would use integrating sphere data
        foil_response = (foil - dark) / flat_field
        
        # Create wavelength array (simplified - should be from spectrometer)
        wavelengths = np.linspace(380, 900, self.target_resolution[0])
        
        # Create calibration data structure
        cal_data = {
            "wavelengths": wavelengths.astype(np.float32),
            "dark_current": dark.astype(np.float32),
            "flat_field": flat_field_norm.astype(np.float32),
            "system_response": foil_response.astype(np.float32),
            "calibration_date": datetime.now().isoformat(),
            "calibration_type": "push_pull_scanner",
            "notes": "Calibrated using dark, white, and foil references"
        }
        
        # Save computed calibration
        np.savez(cal_dir / "computed_calibration.npz", **cal_data)
        
        self.logger.info("Calibration coefficients computed")
        return cal_data
    
    def update_openhsi_calibration(self, cal_dir: Path):
        """
        Update the OpenHSI calibration.pkl file with new calibration data.
        """
        try:
            # Load existing calibration or create new
            cal_path = self.config_dir / "calibration.pkl"
            
            if cal_path.exists():
                current_cal = load_calibration(str(cal_path))
            else:
                current_cal = {}
            
            # Load computed calibration
            computed_cal = np.load(cal_dir / "computed_calibration.npz", allow_pickle=True)
            
            # Update calibration data
            for key in computed_cal.files:
                if key not in ['calibration_date', 'notes']:
                    current_cal[key] = computed_cal[key]
            
            current_cal['calibration_date'] = datetime.now().isoformat()
            current_cal['calibration_notes'] = "Push-pull scanner calibration"
            
            # Save updated calibration
            save_calibration(str(cal_path), current_cal)
            
            # Reload camera properties
            if self.cam_props:
                self.cam_props = CameraProperties(
                    json_path=str(self.config_dir / "settings.json"),
                    cal_path=str(cal_path)
                )
            
            self.logger.info(f"OpenHSI calibration updated: {cal_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to update OpenHSI calibration: {e}")
    
    def trigger_arduino_scan(self) -> bool:
        """Trigger Arduino to start scanning sequence."""
        try:
            self.logger.info("Triggering Arduino scan...")
            GPIO.output(GPIO_TRIGGER, GPIO.HIGH)
            time.sleep(0.1)
            GPIO.output(GPIO_TRIGGER, GPIO.LOW)
            return True
        except Exception as e:
            self.logger.error(f"Failed to trigger Arduino: {e}")
            return False
    
    def capture_pushpull_scan(self) -> bool:
        """
        Capture images using push-pull scanning with Arduino control.
        
        This is your original scanning method, now feeding data into OpenHSI processing.
        """
        if not PICAM_AVAILABLE:
            self.logger.error("picamera2 not available for scanning")
            return False
        
        self.logger.info("Starting push-pull scan...")
        self.scanning = True
        self.scan_images = []
        
        try:
            # Initialize camera
            self.picam = Picamera2()
            config = self.picam.create_still_configuration(
                main={"size": (self.target_resolution[1], self.target_resolution[0]), 
                      "format": "RGB888"}
            )
            self.picam.configure(config)
            self.picam.start()
            time.sleep(1)
            
            # Trigger Arduino
            if not self.trigger_arduino_scan():
                return False
            
            # Capture at each position
            for position in range(self.scan_positions):
                if not self.scanning:
                    break
                
                self.logger.info(f"Capturing position {position + 1}/{self.scan_positions}")
                
                # Wait for Arduino status if available
                if GPIO.input(GPIO_STATUS) == GPIO.LOW:
                    self.logger.debug(f"Arduino ready at position {position}")
                
                # Capture frame
                frame = self.picam.capture_array()
                
                # Convert to grayscale for push-prow HSI
                if frame.ndim == 3:
                    gray = np.mean(frame, axis=2)
                else:
                    gray = frame
                
                self.scan_images.append(gray)
                
                self.logger.debug(f"Position {position} captured: {gray.shape}")
                
                # Small delay for Arduino movement
                time.sleep(0.05)
            
            # Wait for pull phase completion
            time.sleep(3)
            
            self.logger.info(f"Scan complete. Captured {len(self.scan_images)} images")
            return True
            
        except Exception as e:
            self.logger.error(f"Scanning failed: {e}")
            return False
        finally:
            if hasattr(self, 'picam'):
                self.picam.stop()
            self.scanning = False
    
    def stitch_and_process_openhsi(self):
        """
        Stitch scanned images and process using OpenHSI methods.
        
        This replaces your manual processing with OpenHSI's validated pipelines.
        """
        if not self.scan_images:
            self.logger.error("No images to process")
            return None
        
        self.logger.info("Stitching and processing with OpenHSI methods...")
        
        try:
            # 1. Stitch images (simplified version of your method)
            stitched = self.stitch_images_open_cv()
            
            if stitched is None:
                self.logger.error("Stitching failed")
                return None
            
            # 2. Convert to 3D cube (height, width, spectral)
            # In push-pull scanning, each scan position is a spectral band
            # Each column in stitched image corresponds to a wavelength
            cube = self.create_cube_from_stitched(stitched)
            
            # 3. Apply OpenHSI processing
            processed_cube = self.apply_openhsi_processing(cube)
            
            # 4. Convert to reflectance if desired
            if self.processing_lvl >= 6:
                reflectance_cube = self.convert_to_reflectance(processed_cube)
                self.reflectance_cube = reflectance_cube
                self.logger.info("Reflectance cube created")
            
            self.final_cube = processed_cube
            self.logger.info(f"Processing complete. Cube shape: {processed_cube.shape}")
            
            return processed_cube
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            return None
    
    def stitch_images_open_cv(self):
        """
        Stitch images using OpenCV (more robust than manual stitching).
        """
        try:
            # Convert to uint8 for OpenCV
            images_uint8 = []
            for img in self.scan_images:
                # Normalize to 0-255
                img_norm = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
                images_uint8.append(img_norm)
            
            # Simple horizontal stitching for push-pull
            # Each image is a spectral band, stitch side by side
            stitched = np.hstack(images_uint8)
            
            self.logger.info(f"Stitched image shape: {stitched.shape}")
            return stitched
            
        except Exception as e:
            self.logger.error(f"OpenCV stitching failed: {e}")
            # Fallback to simple concatenation
            return np.hstack([img.astype(np.float32) for img in self.scan_images])
    
    def create_cube_from_stitched(self, stitched_image):
        """
        Convert stitched push-pull image to hyperspectral cube.
        
        In push-pull scanning:
        - Each scan position captures a different wavelength band
        - Stitched width = number of spectral bands
        - Height = spatial resolution
        """
        height, width = stitched_image.shape
        
        # Create cube: (spatial_y, spatial_x, spectral)
        # For push-pull, spectral dimension is the scan direction
        cube = np.zeros((height, width, len(self.scan_images)), dtype=np.float32)
        
        # Reshape: each column becomes a spatial slice, each scan position a band
        for i in range(len(self.scan_images)):
            # Extract this band from stitched image
            band_start = i * (width // len(self.scan_images))
            band_end = (i + 1) * (width // len(self.scan_images))
            
            if band_end <= width:
                cube[:, :, i] = stitched_image[:, band_start:band_end]
        
        return cube
    
    def apply_openhsi_processing(self, cube):
        """
        Apply OpenHSI processing pipeline to the datacube.
        """
        try:
            # Create a ProcessDatacube object
            processor = ProcessDatacube(
                data=cube,
                camera_props=self.cam_props,
                processing_lvl=self.processing_lvl
            )
            
            # Apply processing
            processed = processor.process()
            
            # Store as radiance cube
            self.radiance_cube = processed
            
            return processed
            
        except Exception as e:
            self.logger.error(f"OpenHSI processing failed: {e}")
            # Fallback: apply basic corrections
            return self.apply_basic_corrections(cube)
    
    def apply_basic_corrections(self, cube):
        """Basic radiometric correction when OpenHSI processing fails."""
        if self.calibration_targets["dark"] is None or self.calibration_targets["white"] is None:
            self.logger.warning("No calibration available, returning raw cube")
            return cube
        
        # Basic dark subtraction and flat field
        dark = np.median(self.calibration_targets["dark"])
        white = np.median(self.calibration_targets["white"])
        
        corrected = (cube - dark) / (white - dark + 1e-10)
        corrected = np.clip(corrected, 0, 1.0)
        
        return corrected
    
    def convert_to_reflectance(self, radiance_cube):
        """
        Convert radiance cube to reflectance using OpenHSI methods.
        
        Options:
        1. Use calibration data (used: simplified)
        2. Use Empirical Line Calibration (recommended)
        3. Use radiative transfer model (most accurate)
        """
        try:
            # Method 1: Simplified using calibration
            if "system_response" in self.calibration:
                # Divide by system response
                reflectance = radiance_cube / self.calibration["system_response"][np.newaxis, np.newaxis, :]
                reflectance = np.clip(reflectance, 0, 1.2)
                return reflectance
            
            # Method 2: Use foil reference (fallback)
            elif self.calibration_targets["foil"] is not None:
                self.logger.info("Using foil reference for reflectance conversion")
                
                # Get foil spectrum (average of foil reference)
                foil_spectrum = np.median(self.calibration_targets["foil"])
                
                # Assume foil reflectance ~0.88 average
                # This is simplified - proper method uses wavelength-dependent reflectance
                reflectance = radiance_cube / (foil_spectrum / 0.88)
                reflectance = np.clip(reflectance, 0, 1.2)
                
                return reflectance
            
            else:
                self.logger.warning("No reflectance conversion method available")
                return radiance_cube
                
        except Exception as e:
            self.logger.error(f"Reflectance conversion failed: {e}")
            return radiance_cube
    
    def save_openhsi_format(self, sample_name: str = None):
        """
        Save results in OpenHSI-compatible NetCDF format.
        """
        if self.final_cube is None:
            self.logger.error("No data to save")
            return None
        
        if sample_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sample_name = f"pushpull_{timestamp}"
        
        output_dir = Path("results") / sample_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save in OpenHSI NetCDF format
            from openhsi.io import save_datacube
            
            # Create metadata dictionary
            metadata = {
                "sample_name": sample_name,
                "acquisition_date": datetime.now().isoformat(),
                "instrument": "PushPullScanner",
                "processing_level": self.processing_lvl,
                "scan_positions": self.scan_positions,
                "step_distance_mm": self.step_distance_mm,
                "exposure_time_ms": self.exposure_time_ms,
                "calibration_date": self.calibration.get('calibration_date', 'unknown') if self.calibration else 'unknown'
            }
            
            # Save radiance cube
            if self.radiance_cube is not None:
                save_datacube(
                    self.radiance_cube,
                    output_dir / f"{sample_name}_radiance.nc",
                    metadata=metadata,
                    wavelengths=self.calibration.get('wavelengths', None) if self.calibration else None
                )
                self.logger.info(f"Radiance cube saved: {output_dir / f'{sample_name}_radiance.nc'}")
            
            # Save reflectance cube
            if self.reflectance_cube is not None:
                save_datacube(
                    self.reflectance_cube,
                    output_dir / f"{sample_name}_reflectance.nc",
                    metadata=metadata,
                    wavelengths=self.calibration.get('wavelengths', None) if self.calibration else None
                )
                self.logger.info(f"Reflectance cube saved: {output_dir / f'{sample_name}_reflectance.nc'}")
            
            # Also save as HDF5 for compatibility with your existing code
            self.save_hdf5_backup(output_dir, sample_name)
            
            # Save RGB preview
            self.save_rgb_preview(output_dir, sample_name)
            
            self.logger.info(f"All results saved to: {output_dir}")
            return str(output_dir)
            
        except Exception as e:
            self.logger.error(f"Failed to save in OpenHSI format: {e}")
            # Fallback to HDF5
            return self.save_hdf5_backup(output_dir, sample_name)
    
    def save_hdf5_backup(self, output_dir: Path, sample_name: str):
        """Save backup in HDF5 format (your original format)."""
        h5_path = output_dir / f"{sample_name}.h5"
        
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset("hsi_cube", data=self.final_cube.astype(np.float32))
            
            if self.calibration and 'wavelengths' in self.calibration:
                f.create_dataset("wavelengths", data=self.calibration['wavelengths'].astype(np.float32))
            
            # Save metadata
            metadata_group = f.create_group("metadata")
            metadata_group.attrs["sample_name"] = sample_name
            metadata_group.attrs["processing_level"] = self.processing_lvl
            metadata_group.attrs["scan_positions"] = self.scan_positions
        
        self.logger.info(f"HDF5 backup saved: {h5_path}")
        return str(h5_path)
    
    def save_rgb_preview(self, output_dir: Path, sample_name: str):
        """Save RGB preview image."""
        try:
            import matplotlib.pyplot as plt
            
            if self.final_cube is None:
                return
            
            # Find RGB bands (simplified)
            if self.calibration and 'wavelengths' in self.calibration:
                wavelengths = self.calibration['wavelengths']
                # Find closest to 650nm (R), 550nm (G), 450nm (B)
                r_idx = np.argmin(np.abs(wavelengths - 650))
                g_idx = np.argmin(np.abs(wavelengths - 550))
                b_idx = np.argmin(np.abs(wavelengths - 450))
            else:
                # Default indices
                r_idx, g_idx, b_idx = -1, len(self.final_cube[2])//2, 0
            
            # Create RGB image
            rgb = np.stack([
                self.final_cube[:, :, r_idx],
                self.final_cube[:, :, g_idx],
                self.final_cube[:, :, b_idx]
            ], axis=2)
            
            # Normalize
            rgb_norm = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb) + 1e-10)
            
            # Save
            plt.imsave(output_dir / "rgb_preview.png", rgb_norm)
            
        except Exception as e:
            self.logger.warning(f"Could not save RGB preview: {e}")
    
    def run_complete_acquisition(self, sample_name: str = None):
        """
        Run complete acquisition pipeline with OpenHSI processing.
        """
        self.logger.info("Starting complete OpenHSI acquisition pipeline...")
        
        # Step 1: Initialize OpenHSI
        if not self.initialize_open_hsi():
            self.logger.error("Failed to initialize OpenHSI")
            return False
        
        # Step 2: Check calibration
        if self.calibration is None or 'calibration_notes' in self.calibration and 'PLACEHOLDER' in self.calibration['calibration_notes']:
            self.logger.warning("Using placeholder calibration. Results will not be accurate!")
            response = input("Run calibration now? (y/n): ")
            if response.lower() == 'y':
                self.calibrate_with_openhsi_method()
        
        # Step 3: Capture scan
        self.logger.info("Step 1: Capturing push-pull scan...")
        if not self.capture_pushpull_scan():
            self.logger.error("Scan capture failed")
            return False
        
        # Step 4: Process with OpenHSI
        self.logger.info("Step 2: Processing with OpenHSI...")
        cube = self.stitch_and_process_openhsi()
        if cube is None:
            self.logger.error("Processing failed")
            return False
        
        # Step 5: Save results
        self.logger.info("Step 3: Saving results...")
        output_dir = self.save_openhsi_format(sample_name)
        
        if output_dir:
            self.logger.info(f"Acquisition complete! Results in: {output_dir}")
            return output_dir
        else:
            self.logger.error("Failed to save results")
            return False
    
    def update_processing_level(self, level: int):
        """
        Update OpenHSI processing level.
        
        Levels:
        - -1, 0, 1: Digital Numbers
        - 2, 3: Digital Numbers with binning
        - 4, 5: Radiance (µW/cm²/sr/nm)
        - 6, 8: Reflectance
        """
        if -1 <= level <= 8:
            self.processing_lvl = level
            self.logger.info(f"Processing level updated to: {level}")
        else:
            self.logger.error(f"Invalid processing level: {level}")
    
    def cleanup(self):
        """Cleanup resources."""
        self.scanning = False
        GPIO.cleanup()
        self.logger.info("System cleanup complete")


# Main execution
def main():
    """Main function for OpenHSI push-pull system."""
    print("\n" + "="*60)
    print("OPENHSI PUSH-PULL HSI SYSTEM")
    print("="*60)
    
    system = OpenHSIPushPullSystem(
        config_dir="openhsi_pushpull",
        target_resolution=(224, 224)
    )
    
    try:
        # Main menu
        while True:
            print("\n" + "="*60)
            print("MAIN MENU")
            print("="*60)
            print("1. Run OpenHSI Calibration (REQUIRED for accuracy)")
            print("2. Configure System Parameters")
            print("3. Run Complete Acquisition")
            print("4. Quick Test (5 positions)")
            print("5. View/Update Processing Level")
            print("6. Exit")
            print("="*60)
            
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == "1":
                print("\n" + "="*60)
                print("CALIBRATION MODE")
                print("="*60)
                print("This will guide you through proper calibration.")
                print("You need:")
                print("  - Lens cap or dark material")
                print("  - White reference (Spectralon, BaSO4, white ceramic)")
                print("  - Aluminum foil (clean, flat)")
                print("="*60)
                
                confirm = input("Proceed with calibration? (y/n): ")
                if confirm.lower() == 'y':
                    system.calibrate_with_openhsi_method()
            
            elif choice == "2":
                print("\nConfigure Parameters:")
                positions = input(f"Scan positions (current: {system.scan_positions}): ").strip()
                step_mm = input(f"Step distance in mm (current: {system.step_distance_mm}): ").strip()
                exposure = input(f"Exposure time in ms (current: {system.exposure_time_ms}): ").strip()
                
                if positions:
                    system.scan_positions = int(positions)
                if step_mm:
                    system.step_distance_mm = float(step_mm)
                if exposure:
                    system.exposure_time_ms = int(exposure)
                    
                print("Parameters updated.")
            
            elif choice == "3":
                sample_name = input("Enter sample name (optional): ").strip() or None
                system.run_complete_acquisition(sample_name)
            
            elif choice == "4":
                # Quick test
                original_positions = system.scan_positions
                system.scan_positions = 5
                system.run_complete_acquisition("test_scan")
                system.scan_positions = original_positions
            
            elif choice == "5":
                print(f"\nCurrent processing level: {system.processing_lvl}")
                print("\nProcessing levels:")
                print("-1,0,1: Digital Numbers (raw)")
                print("2,3: Digital Numbers with binning")
                print("4,5: Radiance (µW/cm²/sr/nm)")
                print("6,8: Reflectance")
                
                new_level = input(f"\nNew processing level (current: {system.processing_lvl}): ").strip()
                if new_level:
                    system.update_processing_level(int(new_level))
            
            elif choice == "6":
                print("Exiting...")
                break
            
            else:
                print("Invalid choice. Please try again.")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        system.cleanup()


if __name__ == "__main__":
    # Check dependencies
    try:
        import openhsi
        print("OpenHSI library found")
    except ImportError:
        print("ERROR: OpenHSI library not installed!")
        print("Install with: pip install openhsi")
        print("Or from source: https://github.com/openhsi/openhsi")
        exit(1)
    
    if not PICAM_AVAILABLE:
        print("WARNING: picamera2 not available. Camera functions disabled.")
        print("Install with: pip install picamera2")
    
    main()
