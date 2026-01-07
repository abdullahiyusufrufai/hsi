# test_raw_snapshot.py
import time
import numpy as np
from picamera2 import Picamera2
from datetime import datetime

def main():
    print("📸 Testing Pi Camera V2 RAW capture (2 frames)")
    
    # Initialize camera
    picam = Picamera2()
    
    # Configure for RAW capture at max resolution (or close)
    config = picam.create_still_configuration(
        raw={"size": (640, 480)},  # Adjust if you know your sensor limits
        controls={
            "ExposureTime": 10000,   # 10,000 µs = 10 ms
            "AnalogueGain": 1.0,
            "AeEnable": False        # Disable auto-exposure
        }
    )
    picam.configure(config)
    picam.start()
    time.sleep(1)  # Let camera stabilize

    try:
        for i in range(2):
            print(f"\n--- Capturing frame {i+1} ---")
            
            # Capture RAW frame
            raw_frame = picam.capture_array("raw")
            
            # File naming
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:20]
            filename = f"test_frame_{i+1}_{timestamp}.npy"
            
            # Save
            np.save(filename, raw_frame)
            print(f"✅ Saved: {filename}")
            
            # Print metadata
            print(f"  Shape: {raw_frame.shape}")
            print(f"  Data type: {raw_frame.dtype}")
            print(f"  Min/Max: {raw_frame.min()} / {raw_frame.max()}")
            print(f"  Mean: {raw_frame.mean():.1f}")
            
            # Optional: show a tiny sample
            print(f"  Top-left 3x3 pixels:\n{raw_frame[:3, :3]}")
            
            time.sleep(0.5)  # Small gap between frames

        # Get full camera metadata (includes sensor info)
        metadata = picam.capture_metadata()
        print("\n🔍 Full Camera Metadata:")
        for key, val in metadata.items():
            print(f"  {key}: {val}")
            
    finally:
        picam.stop()
        picam.close()
    
    print("\n✅ Test complete! Check .npy files and metadata above.")

if __name__ == "__main__":
    main()
