import argparse
from callibrator import Callibrator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # for calibration:
    parser.add_argument("-i", "--input_dir", type=str, default="./images", help="Directory of the images to be undistorted")
    parser.add_argument("-s", "--save", type=bool, default=False, help="Whether to save the calibration results")
    parser.add_argument("-c", "--calibration_save_path", type=str, default="./calibration.json", help="Path to the calibration file")
    # for undistortion:
    parser.add_argument("-l", "--calibration_load_path", type=str, default=None, help="Path to the calibration file")
    parser.add_argument("-o", "--output_dir", type=str, default="images_undistorted",help="Directory of the undistorted images to be saved in")

    args = parser.parse_args()

    callibrator = Callibrator()

    if args.calibration_load_path is not None:
        callibrator.load_calibration(args.calibration_load_path)
    else:
        callibrator.calibrate(args.input_dir)
    
    if args.save:
        callibrator.save_calibration(args.calibration_save_path)

    # undistort
    callibrator.undistort(args.input_dir, args.save, args.output_dir)