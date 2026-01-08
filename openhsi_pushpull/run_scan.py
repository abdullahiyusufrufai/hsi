#!/usr/bin/env python3
"""
Automated Push-Pull HSI Scan Launcher

This script guides you through synchronized scan acquisition.
Since Arduino runs independently, you will manually reset it at the right time.
"""

import sys
import time
import argparse
import subprocess
import logging
from pathlib import Path


def setup_basic_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def countdown(seconds: int, message: str):
    """Display a countdown with a message."""
    print(f"\n{message}")
    for i in range(seconds, 0, -1):
        print(f"  Starting in {i}...", end="\r")
        time.sleep(1)
    print("  Starting now!           ")


def run_capture(
    sample_name: str,
    scans: int,
    delay: float,
    height: int,
    bands: int,
    exposure: int,
    mock: bool = False
):
    """Launch the main acquisition CLI as a subprocess."""
    cmd = [
        sys.executable, "-m", "openhsi_pushpull.cli",
        "--sample", sample_name,
        "--scans", str(scans),
        "--delay", str(delay),
        "--height", str(height),
        "--bands", str(bands),
        "--exposure", str(exposure),
    ]
    if mock:
        cmd.append("--mock")

    print("\n🚀 Launching capture process...\n")
    try:
        result = subprocess.run(cmd, check=True)
        if result.returncode == 0:
            print(f"\n✅ Success! Data saved in: results/{sample_name}/")
        else:
            print("\n❌ Capture failed.")
    except subprocess.CalledProcessError as e:
        print(f"\n💥 Capture process crashed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Capture interrupted by user.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Automated HSI Scan Launcher")
    parser.add_argument("--sample", type=str, help="Sample name (auto-generated if omitted)")
    parser.add_argument("--scans", type=int, default=100, help="Number of scan lines (default: 100)")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between lines in seconds (default: 0.5)")
    parser.add_argument("--height", type=int, default=512, help="Spatial height (default: 512)")
    parser.add_argument("--bands", type=int, default=256, help="Spectral bands (default: 256)")
    parser.add_argument("--exposure", type=int, default=100000, help="Exposure time in µs (default: 100000)")
    parser.add_argument("--mock", action="store_true", help="Use mock data (no camera needed)")
    parser.add_argument("--auto", action="store_true", help="Skip instructions (for headless use)")

    args = parser.parse_args()

    setup_basic_logging()

    # Auto-generate sample name if not provided
    if not args.sample:
        args.sample = f"scan_{time.strftime('%Y%m%d_%H%M%S')}"

    print("=" * 60)
    print("AUTOMATED PUSH-PULL HSI SCAN")
    print("=" * 60)
    print(f"Sample name:     {args.sample}")
    print(f"Scan lines:      {args.scans}")
    print(f"Delay/line:      {args.delay} s")
    print(f"Resolution:      {args.height} (spatial) x {args.bands} (spectral)")
    print(f"Exposure:        {args.exposure} µs")
    print("=" * 60)

    if not args.auto and not args.mock:
        print("\n🔧 HARDWARE INSTRUCTIONS:")
        print("1. Ensure your Arduino is powered and ready.")
        print("2. When prompted, PRESS THE ARDUINO RESET BUTTON.")
        print("   → This starts its scanning motion loop.")
        print("3. The Pi will begin capturing 2 seconds after reset.")
        print("\n⚠️  Timing is critical! Be ready to press reset.\n")

        input("✅ Press ENTER when you're ready to begin...")

        countdown(3, "👉 PRESS ARDUINO RESET BUTTON NOW!")

        # Wait for Arduino to start moving (adjust based on your code)
        countdown(2, "Waiting for stage to stabilize...")

    elif args.mock:
        print("\n🧪 Running in MOCK mode (no hardware needed).")
        time.sleep(1)

    # Launch capture
    run_capture(
        sample_name=args.sample,
        scans=args.scans,
        delay=args.delay,
        height=args.height,
        bands=args.bands,
        exposure=args.exposure,
        mock=args.mock
    )


if __name__ == "__main__":
    main()
