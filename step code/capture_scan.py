# capture_scan.py
import time
import numpy as np
from pathlib import Path
from picamera2 import Picamera2

# ===== CONFIGURATION =====
OUTPUT_DIR = Path("scans/scan_{}".format(time.strftime("%Y%m%d_%H%M%S")))
OUTPUT_DIR.mkdir(parents=True)

# Optical ROI: adjust after test capture!
SLIT_X_START = 100   # left edge of slit image
SLIT_X_END = 540     # right edge
SPECTRAL_HEIGHT = 480  # full height = spectral axis

# Scanning parameters
NUM_LINES = 50        # match your mechanical travel
CAPTURE_INTERVAL_S = 0.2  # e.g., 5 FPS → stage must move 1mm per 0.2s

# =========================

def main():
    print(f"Starting scan: {NUM_LINES} lines, interval={CAPTURE_INTERVAL_S}s")
    print(f"Saving to: {OUTPUT_DIR}")

    # Initialize camera in RAW mode
    picam = Picamera2()
    config = picam.create_still_configuration(
        raw={"size": (640, SPECTRAL_HEIGHT)},  # full sensor width, spectral height
        controls={
            "ExposureTime": 10000,   # 10 ms (adjust for lighting)
            "AnalogueGain": 1.0,
            "FrameRate": 10.0
        }
    )
    picam.configure(config)
    picam.start()
    time.sleep(1)  # warm-up

    try:
        for i in range(NUM_LINES):
            print(f"Capturing line {i+1}/{NUM_LINES}")
            raw_frame = picam.capture_array("raw")  # shape: (480, 640), dtype: uint16

            # Crop to slit region → (480, SLIT_WIDTH)
            cropped = raw_frame[:, SLIT_X_START:SLIT_X_END]

            # Save immediately to avoid RAM overflow
            np.save(OUTPUT_DIR / f"line_{i:04d}.npy", cropped)

            time.sleep(CAPTURE_INTERVAL_S)

        print(f"✅ Scan complete! Data saved to: {OUTPUT_DIR}")

    finally:
        picam.stop()
        picam.close()

if __name__ == "__main__":
    main()
