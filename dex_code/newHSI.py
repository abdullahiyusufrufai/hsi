import os
import time
import numpy as np
import cv2
import h5py
import json
import logging
from datetime import datetime
from picamera2 import Picamera2
import RPi.GPIO as GPIO
from scipy import interpolate, signal
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import threading
import queue

# GPIO Configuration
GPIO_TRIGGER = 17     # Raspberry Pi GPIO to trigger Arduino
GPIO_STATUS = 27      # Arduino status pin (optional)
GPIO.setmode(GPIO.BCM)
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_STATUS, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Foil Reflectance Table
FOIL_REFLECTANCE_TABLE = {
    380: 0.88, 400: 0.89, 420: 0.90, 440: 0.91, 460: 0.92, 480: 0.92, 500: 0.93,
    520: 0.93, 540: 0.94, 560: 0.94, 580: 0.94, 600: 0.93, 620: 0.92, 640: 0.91,
    660: 0.90, 680: 0.89, 700: 0.88, 720: 0.87, 740: 0.86, 760: 0.85, 780: 0.84,
    800: 0.83, 820: 0.82, 840: 0.81, 860: 0.80, 880: 0.79, 900: 0.78,
}

FOIL_WL = np.array(list(FOIL_REFLECTANCE_TABLE.keys()))
FOIL_REFLECT = np.array(list(FOIL_REFLECTANCE_TABLE.values()))

class PushPullHSISystem:
    def __init__(self, target_resolution=(224, 224)):
        self.target_resolution = target_resolution
        self.picam = None
        self.camera_initialized = False
        
        # Scanning parameters
        self.scan_positions = 10
        self.step_distance_mm = 0.1
        self.exposure_time_ms = 5000
        self.overlap_threshold = 0.8
        
        # Calibration data
        self.dark_ref = None
        self.white_ref = None
        self.foil_ref = None
        self.wavelength_map = None
        self.mapping_a = None
        self.mapping_b = None
        
        # Image stitching buffers
        self.scan_images = []
        self.current_position = 0
        
        # Quality control
        self.quality_threshold = 0.85
        self.max_retries = 3
        
        # Threading for synchronization
        self.capture_queue = queue.Queue()
        self.scanning = False
        
        # Results storage
        self.final_cube = None
        self.stitched_images = {}
        
        # Target wavelengths (380-900nm in 10nm steps)
        self.target_wavelengths = np.arange(380, 901, 10) #53 bands
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('pushpull_hsi.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        print("\n" + "="*60)
        print("PUSH-PULL HYPERSPECTRAL IMAGING SYSTEM")
        print("="*60)
        print(f"Target resolution: {target_resolution}")
        print(f"Scan positions: {self.scan_positions}")
        print(f"Step distance: {self.step_distance_mm} mm")
        print(f"Exposure time: {self.exposure_time_ms} ms")
        
    def initialize_camera(self, resolution=(1920, 1080)):
        """Initialize camera with thermal stabilization."""
        try:
            self.picam = Picamera2()
            config = self.picam.create_still_configuration(
                main={"size": resolution, "format": "RGB888"},
                raw={"size": resolution}
            )
            self.picam.configure(config)
            self.picam.start()
            
            # Thermal stabilization
            self.logger.info("Thermal stabilization (30 seconds)...")
            time.sleep(30)
            
            self.camera_initialized = True
            self.logger.info("Camera initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Camera initialization failed: {e}")
            return False
    
    def trigger_arduino_scan(self):
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
    
    def check_image_quality(self, image):
        """Check image quality using multiple metrics."""
        if image is None:
            return 0.0
        
        # Convert to grayscale if RGB
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Calculate quality metrics
        metrics = {}
        
        # 1. Sharpness (using Laplacian variance)
        metrics['sharpness'] = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var()
        
        # 2. Illumination uniformity
        h, w = gray.shape
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        border_region = np.concatenate([
            gray[:h//4, :].flatten(),
            gray[3*h//4:, :].flatten(),
            gray[:, :w//4].flatten(),
            gray[:, 3*w//4:].flatten()
        ])
        metrics['uniformity'] = 1 - (abs(np.mean(center_region) - np.mean(border_region)) / 
                                    (np.mean(gray) + 1e-8))
        
        # 3. Signal-to-Noise Ratio (simplified)
        signal = np.mean(gray)
        noise = np.std(gray)
        metrics['snr'] = signal / (noise + 1e-8)
        
        # 4. Contrast
        metrics['contrast'] = (np.max(gray) - np.min(gray)) / (np.max(gray) + np.min(gray) + 1e-8)
        
        # Combined quality score
        quality_score = (
            0.3 * np.clip(metrics['sharpness'] / 1000, 0, 1) +
            0.3 * metrics['uniformity'] +
            0.2 * np.clip(metrics['snr'] / 100, 0, 1) +
            0.2 * metrics['contrast']
        )
        
        self.logger.debug(f"Quality metrics: {metrics}")
        self.logger.debug(f"Quality score: {quality_score:.3f}")
        
        return quality_score
    
    def capture_with_retry(self, position, max_retries=3):
        """Capture image with automatic retry on quality failure."""
        for attempt in range(max_retries):
            try:
                # Wait for Arduino to be ready (optional status pin)
                if GPIO.input(GPIO_STATUS) == GPIO.LOW:
                    self.logger.info(f"Arduino ready for position {position}")
                
                # Capture image
                raw_image = self.picam.capture_array()
                image = cv2.resize(raw_image, 
                                  (self.target_resolution[1], self.target_resolution[0]))
                
                # Check quality
                quality = self.check_image_quality(image)
                
                if quality >= self.quality_threshold:
                    self.logger.info(f"Position {position}: Quality OK ({quality:.3f})")
                    return image, quality
                else:
                    self.logger.warning(f"Position {position}: Quality low ({quality:.3f}), retry {attempt + 1}")
                    
            except Exception as e:
                self.logger.error(f"Capture failed at position {position}: {e}")
            
            time.sleep(1)  # Brief pause before retry
        
        self.logger.error(f"Failed to capture quality image at position {position}")
        return None, 0.0
    
    def run_push_pull_scan(self):
        """Execute complete push-pull scanning sequence."""
        if not self.camera_initialized:
            self.logger.error("Camera not initialized")
            return False
        
        self.logger.info("Starting push-pull scanning sequence...")
        self.scanning = True
        self.scan_images = []
        capture_qualities = []
        
        try:
            # Trigger Arduino to start scanning
            if not self.trigger_arduino_scan():
                return False
            
            # Capture images at each position
            for position in range(self.scan_positions):
                if not self.scanning:
                    break
                
                self.logger.info(f"Capturing position {position + 1}/{self.scan_positions}")
                
                # Capture with quality check and retry
                image, quality = self.capture_with_retry(position)
                
                if image is not None:
                    self.scan_images.append(image)
                    capture_qualities.append(quality)
                    
                    # Log progress
                    self.logger.info(f"Position {position + 1} captured (Quality: {quality:.3f})")
                    
                    # Wait for next position (handled by Arduino timing)
                    # Platform moves during this time
                else:
                    self.logger.error(f"Failed to capture position {position}")
                    self.scanning = False
                    return False
            
            # Wait for Arduino to complete pull phase
            time.sleep(5)  # Adjust based on your return time
            
            self.logger.info(f"Scan complete. Average quality: {np.mean(capture_qualities):.3f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Scanning failed: {e}")
            self.scanning = False
            return False
    
    def stitch_push_pull_images(self):
        """Stitch push-pull scanned images using OpenHSI-inspired method."""
        if not self.scan_images:
            self.logger.error("No images to stitch")
            return None
        
        self.logger.info(f"Stitching {len(self.scan_images)} images...")
        
        # Convert to grayscale for feature detection
        gray_images = []
        for img in self.scan_images:
            if len(img.shape) == 3:
                gray_images.append(np.mean(img, axis=2))
            else:
                gray_images.append(img)
        
        # Use first image as reference
        reference = gray_images[0]
        H, W = reference.shape
        
        # Initialize stitched image
        total_width = W + (len(gray_images) - 1) * int(W * (1 - self.overlap_threshold))
        stitched = np.zeros((H, total_width), dtype=np.float32)
        
        # Place first image
        stitched[:, :W] = reference
        
        current_x = 0
        
        for i in range(1, len(gray_images)):
            current_img = gray_images[i]
            
            # Feature-based alignment (simplified - use OpenCV for better results)
            # For precise stitching, consider using OpenCV's stitching module
            overlap_width = int(W * self.overlap_threshold)
            
            # Extract overlap regions
            left_region = gray_images[i-1][:, -overlap_width:]
            right_region = current_img[:, :overlap_width]
            
            # Find optimal alignment using cross-correlation
            correlation = signal.correlate2d(left_region, right_region, mode='same')
            max_pos = np.unravel_index(np.argmax(correlation), correlation.shape)
            
            # Calculate offset
            offset_y = max_pos[0] - overlap_width // 2
            offset_x = max_pos[1] - overlap_width // 2
            
            # Calculate placement position
            overlap_used = overlap_width - abs(offset_x)
            placement_x = current_x + W - overlap_used
            
            # Place image with blending
            blend_width = min(50, overlap_used // 2)
            
            if blend_width > 0:
                # Create blend mask
                blend_mask = np.ones((H, W), dtype=np.float32)
                blend_mask[:, :blend_width] = np.linspace(0, 1, blend_width).reshape(1, -1)
                blend_mask[:, -blend_width:] = np.linspace(1, 0, blend_width).reshape(1, -1)
                
                # Blend with existing content
                existing_region = stitched[:, placement_x:placement_x+W]
                if existing_region.shape[1] == W:
                    # Weighted blend
                    blended = existing_region * (1 - blend_mask) + current_img * blend_mask
                    stitched[:, placement_x:placement_x+W] = blended
                else:
                    # No overlap, just place
                    stitched[:, placement_x:placement_x+W] = current_img
            else:
                # No blending, just place
                stitched[:, placement_x:placement_x+W] = current_img
            
            current_x = placement_x
        
        # Crop to actual content
        non_zero_cols = np.where(np.sum(stitched, axis=0) > 0)[0]
        if len(non_zero_cols) > 0:
            stitched = stitched[:, non_zero_cols[0]:non_zero_cols[-1]+1]
        
        self.logger.info(f"Stitching complete. Final dimensions: {stitched.shape}")
        return stitched
    
    def calibrate_system(self):
        """Run calibration once (not during scanning)."""
        self.logger.info("Running system calibration...")
        
        # Initialize camera if not already
        if not self.camera_initialized:
            self.initialize_camera()
        
        # Capture dark reference
        self.logger.info("Capturing dark reference...")
        input("Cover lens and press Enter...")
        dark_frames = []
        for i in range(10):
            frame = self.picam.capture_array()
            dark_frames.append(cv2.resize(frame, 
                                         (self.target_resolution[1], self.target_resolution[0])))
            time.sleep(0.1)
        self.dark_ref = np.mean(dark_frames, axis=0)
        
        # Capture white reference
        self.logger.info("Capturing white reference...")
        input("Place white reference and press Enter...")
        white_frames = []
        for i in range(10):
            frame = self.picam.capture_array()
            white_frames.append(cv2.resize(frame, 
                                          (self.target_resolution[1], self.target_resolution[0])))
            time.sleep(0.1)
        self.white_ref = np.mean(white_frames, axis=0)
        
        # Capture foil reference
        self.logger.info("Capturing foil reference...")
        input("Place aluminum foil and press Enter...")
        foil_frames = []
        for i in range(10):
            frame = self.picam.capture_array()
            foil_frames.append(cv2.resize(frame, 
                                         (self.target_resolution[1], self.target_resolution[0])))
            time.sleep(0.1)
        self.foil_ref = np.mean(foil_frames, axis=0)
        
        # Wavelength calibration (simplified - use your existing method)
        self.mapping_a = 0.7  # nm/pixel
        self.mapping_b = 380  # nm at pixel 0
        self.wavelength_map = self.mapping_a * np.arange(self.target_resolution[1]) + self.mapping_b
        
        self.logger.info("Calibration complete")
        return True
    
    def process_stitched_image(self, stitched_image):
        """Process stitched image to create hyperspectral cube."""
        if stitched_image is None:
            self.logger.error("No stitched image to process")
            return None
        
        self.logger.info("Processing stitched image to hyperspectral cube...")
        
        # Convert to spectral cube using wavelength mapping
        H, W = stitched_image.shape
        bands = len(self.target_wavelengths)
        cube = np.zeros((H, W, bands), dtype=np.float32)
        
        # Create foil reflectance interpolator
        foil_interp = interpolate.interp1d(
            FOIL_WL, FOIL_REFLECT, 
            kind='cubic', bounds_error=False,
            fill_value=(FOIL_REFLECT[0], FOIL_REFLECT[-1])
        )
        
        # Calculate pixel wavelengths (assuming linear mapping across stitched image)
        # This assumes the slit is along the scanning direction
        pixel_wavelengths = np.linspace(
            self.wavelength_map[0], 
            self.wavelength_map[-1], 
            W
        )
        
        # Apply radiometric correction
        eps = 1e-10
        
        # Convert references to grayscale if needed
        if self.dark_ref.ndim == 3:
            dark_gray = np.mean(self.dark_ref, axis=2)
        else:
            dark_gray = self.dark_ref
            
        if self.white_ref.ndim == 3:
            white_gray = np.mean(self.white_ref, axis=2)
        else:
            white_gray = self.white_ref
            
        if self.foil_ref.ndim == 3:
            foil_gray = np.mean(self.foil_ref, axis=2)
        else:
            foil_gray = self.foil_ref
        
        # Take median of references (assuming uniform)
        dark_value = np.median(dark_gray)
        white_value = np.median(white_gray)
        foil_value = np.median(foil_gray)
        
        # Calculate normalized responses
        measured_foil = (foil_value - dark_value) / (white_value - dark_value + eps)
        
        # Process each column (wavelength) in the stitched image
        for x in range(W):
            # Get wavelength for this column
            wl = pixel_wavelengths[x]
            
            # Find closest target wavelength band
            wl_idx = np.argmin(np.abs(self.target_wavelengths - wl))
            
            # Get foil reflectance at this wavelength
            foil_reflectance = foil_interp(wl)
            foil_reflectance = np.clip(foil_reflectance, 0.75, 0.95)
            
            # Calculate system response
            system_response = measured_foil * foil_reflectance
            system_response = np.clip(system_response, 0.5, 1.5)
            
            # Get image column values
            column_values = stitched_image[:, x]
            
            # Apply radiometric correction
            measured_sample = (column_values - dark_value) / (white_value - dark_value + eps)
            
            # Calculate true reflectance
            reflectance = measured_sample / (system_response + eps)
            reflectance = np.clip(reflectance, 0.0, 1.2)
            
            # Assign to cube
            cube[:, x, wl_idx] = reflectance
            
            if x % 100 == 0:
                self.logger.debug(f"Processed column {x}/{W}")
        
        self.logger.info(f"Cube created: {cube.shape}")
        self.final_cube = cube
        return cube
    
    def save_results(self, sample_name=None):
        """Save final stitched results with spectral information."""
        if self.final_cube is None:
            self.logger.error("No data to save")
            return None
        
        if sample_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sample_name = f"pushpull_sample_{timestamp}"
        
        output_dir = f"results_{sample_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save HDF5 file with all spectral information
        h5_filename = os.path.join(output_dir, f"{sample_name}.h5")
        with h5py.File(h5_filename, 'w') as f:
            # Save hyperspectral cube
            f.create_dataset("hsi_cube", data=self.final_cube.astype(np.float32))
            
            # Save wavelengths
            f.create_dataset("wavelengths", data=self.target_wavelengths.astype(np.float32))
            f.create_dataset("wavelength_map", data=self.wavelength_map.astype(np.float32))
            
            # Save calibration parameters
            calib_group = f.create_group("calibration")
            calib_group.attrs["mapping_a"] = self.mapping_a
            calib_group.attrs["mapping_b"] = self.mapping_b
            calib_group.attrs["calibration_date"] = datetime.now().isoformat()
            
            # Save scanning parameters
            scan_group = f.create_group("scanning")
            scan_group.attrs["positions"] = self.scan_positions
            scan_group.attrs["step_distance_mm"] = self.step_distance_mm
            scan_group.attrs["exposure_time_ms"] = self.exposure_time_ms
            
            # Save metadata
            meta_group = f.create_group("metadata")
            meta_group.attrs["sample_name"] = sample_name
            meta_group.attrs["resolution"] = str(self.target_resolution)
            meta_group.attrs["creation_time"] = datetime.now().isoformat()
        
        # Save individual band images
        for i, wl in enumerate(self.target_wavelengths):
            band_image = self.final_cube[:, :, i]
            
            # Normalize for visualization
            if np.max(band_image) > np.min(band_image):
                normalized = (band_image - np.min(band_image)) / (np.max(band_image) - np.min(band_image))
            else:
                normalized = band_image
            
            # Save as PNG
            png_filename = os.path.join(output_dir, f"band_{int(wl)}nm.png")
            plt.imsave(png_filename, normalized, cmap='gray')
            
            # Save as numpy array
            np_filename = os.path.join(output_dir, f"band_{int(wl)}nm.npy")
            np.save(np_filename, band_image)
        
        # Save summary image
        rgb_indices = [
            np.argmin(np.abs(self.target_wavelengths - 650)),  # Red
            np.argmin(np.abs(self.target_wavelengths - 550)),  # Green
            np.argmin(np.abs(self.target_wavelengths - 450)),  # Blue
        ]
        
        rgb_image = np.stack([
            self.final_cube[:, :, rgb_indices[0]],
            self.final_cube[:, :, rgb_indices[1]],
            self.final_cube[:, :, rgb_indices[2]]
        ], axis=2)
        
        # Normalize
        rgb_image = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image))
        plt.imsave(os.path.join(output_dir, "rgb_composite.png"), rgb_image)
        
        self.logger.info(f"Results saved to: {output_dir}")
        return output_dir
    
    def run_complete_acquisition(self, sample_name=None):
        """Run complete acquisition pipeline."""
        self.logger.info("Starting complete acquisition pipeline...")
        
        # Step 1: Ensure calibration is loaded
        if self.dark_ref is None:
            self.logger.warning("No calibration found. Running calibration...")
            self.calibrate_system()
        
        # Step 2: Run push-pull scan
        success = self.run_push_pull_scan()
        if not success:
            self.logger.error("Scanning failed")
            return False
        
        # Step 3: Stitch images
        stitched = self.stitch_push_pull_images()
        if stitched is None:
            self.logger.error("Stitching failed")
            return False
        
        # Step 4: Process to hyperspectral cube
        cube = self.process_stitched_image(stitched)
        if cube is None:
            self.logger.error("Cube processing failed")
            return False
        
        # Step 5: Save results
        output_dir = self.save_results(sample_name)
        
        self.logger.info("Acquisition pipeline complete")
        return output_dir
    
    def update_parameters(self, positions=None, step_mm=None, exposure_ms=None):
        """Update scanning parameters."""
        if positions is not None:
            self.scan_positions = positions
            self.logger.info(f"Scan positions updated to: {positions}")
        
        if step_mm is not None:
            self.step_distance_mm = step_mm
            self.logger.info(f"Step distance updated to: {step_mm} mm")
        
        if exposure_ms is not None:
            self.exposure_time_ms = exposure_ms
            self.logger.info(f"Exposure time updated to: {exposure_ms} ms")
    
    def cleanup(self):
        """Cleanup resources."""
        self.scanning = False
        if self.picam is not None and self.camera_initialized:
            try:
                self.picam.stop()
                self.logger.info("Camera stopped")
            except:
                pass
        GPIO.cleanup()
        self.logger.info("System cleanup complete")

# Main execution
def main():
    """Main function for push-pull HSI system."""
    system = PushPullHSISystem(target_resolution=(224, 224))
    
    try:
        # Initialize camera
        if not system.initialize_camera():
            print("Failed to initialize camera")
            return
        
        # Main menu
        while True:
            print("\n" + "="*60)
            print("PUSH-PULL HSI SYSTEM - MAIN MENU")
            print("="*60)
            print("1. Run Calibration (One-time setup)")
            print("2. Configure Scanning Parameters")
            print("3. Run Complete Push-Pull Acquisition")
            print("4. Quick Test Scan (5 positions)")
            print("5. Exit")
            print("="*60)
            
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == "1":
                system.calibrate_system()
                
            elif choice == "2":
                print("\nConfigure Scanning Parameters:")
                positions = input(f"Number of positions (current: {system.scan_positions}): ").strip()
                step_mm = input(f"Step distance in mm (current: {system.step_distance_mm}): ").strip()
                exposure_ms = input(f"Exposure time in ms (current: {system.exposure_time_ms}): ").strip()
                
                if positions:
                    system.update_parameters(positions=int(positions))
                if step_mm:
                    system.update_parameters(step_mm=float(step_mm))
                if exposure_ms:
                    system.update_parameters(exposure_ms=int(exposure_ms))
                    
            elif choice == "3":
                sample_name = input("Enter sample name (optional): ").strip() or None
                system.run_complete_acquisition(sample_name)
                
            elif choice == "4":
                # Quick test with fewer positions
                original_positions = system.scan_positions
                system.update_parameters(positions=5)
                system.run_complete_acquisition("test_scan")
                system.update_parameters(positions=original_positions)
                
            elif choice == "5":
                print("Exiting...")
                break
                
            else:
                print("Invalid choice")
                
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        system.cleanup()

if __name__ == "__main__":
    main()