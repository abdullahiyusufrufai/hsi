import argparse
import logging
import sys
from pathlib import Path
from openhsi_pushpull.acquisition.scanner import PushBroomScanner
from openhsi_pushpull.calibration.calibrator import Calibrator
from openhsi_pushpull.processing.processor import DataProcessor
from openhsi_pushpull.storage.saver import DataSaver

def setup_logging(config_dir: Path):
    config_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(config_dir / "system.log"),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )

def main():
    parser = argparse.ArgumentParser(description="OpenHSI Push-Pull HSI System")
    parser.add_argument("--config-dir", type=str, default="openhsi_config")
    parser.add_argument("--height", type=int, default=512, help="Spatial height (pixels)")
    parser.add_argument("--bands", type=int, default=256, help="Spectral bands")
    parser.add_argument("--scans", type=int, default=100, help="Number of scan lines")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between captures (s)")
    parser.add_argument("--exposure", type=int, default=100000, help="Exposure time (µs)")
    parser.add_argument("--sample", type=str, help="Sample name")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration only")
    parser.add_argument("--mock", action="store_true", help="Use mock scanner")

    args = parser.parse_args()
    config_dir = Path(args.config_dir)
    setup_logging(config_dir)

    if args.mock:
        from openhsi_pushpull.acquisition.mock_scanner import mock_capture_lines
        scan_lines = mock_capture_lines(args.scans, args.height)
    else:
        scanner = PushBroomScanner(
            spatial_height=args.height,
            spectral_bands=args.bands,
            exposure_time_us=args.exposure,
            step_delay_s=args.delay
        )
        if args.calibrate:
            calibrator = Calibrator(scanner)
            cal_file = calibrator.run_calibration()
            return 0 if cal_file else 1

        scan_lines = scanner.capture_lines(args.scans)
        if not scan_lines:
            logging.error("No scan lines captured")
            return 1

    # Process
    cal_file = Path("calibration/calibration.h5")
    processor = DataProcessor(cal_file if cal_file.exists() else None)
    cube = processor.build_cube(scan_lines, args.bands)
    processed_cube = processor.process(cube)

    # Save
    saver = DataSaver()
    metadata = {
        "sample_name": args.sample or "unnamed",
        "spatial_height": args.height,
        "spatial_width": len(scan_lines),
        "spectral_bands": args.bands,
        "scan_positions": args.scans,
    }
    wavelengths = processor.cal_data["wavelengths"] if processor.cal_data else None
    output = saver.save(processed_cube, args.sample, metadata, wavelengths)
    logging.info(f"Saved to {output}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
