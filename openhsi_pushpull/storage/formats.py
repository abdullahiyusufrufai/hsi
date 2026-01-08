import h5py
from pathlib import Path
from typing import Any

def save_h5_cube(
    output_path: Path,
    cube: np.ndarray,
    metadata: dict,
    wavelengths: Optional[np.ndarray] = None
):
    """Save datacube to HDF5."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("data", data=cube, compression="gzip")
        if wavelengths is not None:
            f.create_dataset("wavelengths", data=wavelengths)
        for key, value in metadata.items():
            f.attrs[key] = str(value)
