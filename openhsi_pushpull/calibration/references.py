import h5py
import numpy as np
from pathlib import Path
from typing import Optional

def save_calibration(
    cal_dir: Path,
    dark_ref: np.ndarray,
    white_ref: np.ndarray,
    wavelengths: np.ndarray
) -> Path:
    """Save calibration data to HDF5."""
    cal_dir.mkdir(exist_ok=True, parents=True)
    cal_file = cal_dir / "calibration.h5"
    
    with h5py.File(cal_file, "w") as f:
        f.create_dataset("dark_reference", data=dark_ref[:, np.newaxis])
        f.create_dataset("white_reference", data=white_ref[:, np.newaxis])
        f.create_dataset("wavelengths", data=wavelengths)
        f.attrs["calibration_date"] = np.string_(time.ctime())
    return cal_file

def load_calibration(cal_file: Path) -> Optional[dict]:
    """Load calibration from HDF5."""
    if not cal_file.exists():
        return None
    with h5py.File(cal_file, "r") as f:
        return {
            "dark": f["dark_reference"][:],
            "white": f["white_reference"][:],
            "wavelengths": f["wavelengths"][:]
        }
