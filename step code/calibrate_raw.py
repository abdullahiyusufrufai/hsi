# calibrate.py
from picamera2 import Picamera2
import numpy as np
from pathlib import Path

Path("calib").mkdir(exist_ok=True)
picam = Picamera2()
config = picam.create_still_configuration(raw={"size": (640, 480)})
picam.configure(config)
picam.start()
time.sleep(1)

# DARK: cover slit completely
input("Cover slit → Press Enter")
dark = picam.capture_array("raw")
np.save("calib/dark_raw.npy", dark)

# WHITE: place Spectralon/white tile under same light
input("Place white reference → Press Enter")
white = picam.capture_array("raw")
np.save("calib/white_raw.npy", white)

picam.stop()
print("✅ Calibration saved to calib/")
