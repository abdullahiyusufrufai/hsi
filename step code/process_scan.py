# process_scan.py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Load calibration (run after capturing dark/white)
DARK = np.load("calib/dark_raw.npy")[:, SLIT_X_START:SLIT_X_END]  # same crop!
WHITE = np.load("calib/white_raw.npy")[:, SLIT_X_START:SLIT_X_END]

# Wavelength calibration (example: linear from 400–850 nm)
# REPLACE THIS with your actual calibration!
SPECTRAL_HEIGHT = DARK.shape[0]
WAVELENGTHS = np.linspace(400, 850, SPECTRAL_HEIGHT).astype(np.float32)

def load_and_correct_scan(scan_dir: str):
    scan_path = Path(scan_dir)
    line_files = sorted(scan_path.glob("line_*.npy"))
    
    # Load first frame to get shape
    first = np.load(line_files[0])
    h, w = first.shape
    num_lines = len(line_files)
    
    print(f"Reconstructing cube: {num_lines} (Y) × {w} (X) × {h} (λ)")
    
    # Pre-allocate cube: (Y, X, λ)
    cube = np.zeros((num_lines, w, h), dtype=np.float32)
    
    for i, f in enumerate(line_files):
        raw = np.load(f)
        # Dark + white correction
        corrected = (raw.astype(np.float32) - DARK) / (WHITE - DARK + 1e-6)
        corrected = np.clip(corrected, 0, 2.0)  # allow >1 for bright spots
        cube[i, :, :] = corrected.T  # transpose: (λ, X) → (X, λ) per line
    
    return cube, WAVELENGTHS

def save_cube(cube, wavelengths, output_path):
    np.savez_compressed(
        output_path,
        hsi_cube=cube,
        wavelengths=wavelengths,
        metadata={
            "spatial_shape": cube.shape[:2],
            "spectral_bands": len(wavelengths),
            "units": "reflectance"
        }
    )
    print(f"💾 Cube saved: {output_path}")

# ===== RUN PROCESSING =====
if __name__ == "__main__":
    SCAN_DIR = "scans/scan_20260106_120000"  # ← CHANGE THIS!
    OUTPUT_FILE = f"{SCAN_DIR}/datacube.npz"
    
    cube, wl = load_and_correct_scan(SCAN_DIR)
    save_cube(cube, wl, OUTPUT_FILE)
    
    # Optional: preview average spectrum
    avg_spec = np.mean(cube, axis=(0,1))
    plt.plot(wl, avg_spec)
    plt.xlabel("Wavelength (nm)"); plt.ylabel("Reflectance")
    plt.title("Average Spectrum")
    plt.savefig(f"{SCAN_DIR}/avg_spectrum.png")
    plt.show()
