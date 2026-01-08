import numpy as np
import time
from typing import List

def mock_capture_lines(num_lines: int, spatial_height: int = 512) -> List[np.ndarray]:
    """Generate synthetic scan lines for testing."""
    lines = []
    for i in range(num_lines):
        # Simulate spectral signature (e.g., Gaussian peaks)
        line = np.random.normal(100, 10, spatial_height).astype(np.float32)
        lines.append(line)
        time.sleep(0.01)  # Simulate delay
    return lines
