from pathlib import Path
from datetime import datetime
from openhsi_pushpull.storage.formats import save_h5_cube

class DataSaver:
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)

    def save(self, cube: np.ndarray, sample_name: str, metadata: dict, wavelengths=None):
        if not sample_name:
            sample_name = f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        h5_path = self.output_dir / sample_name / f"{sample_name}.h5"
        save_h5_cube(h5_path, cube, metadata, wavelengths)
        return str(h5_path)
