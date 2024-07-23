# Fisheye Camera Calibration
A tiny callibrator for fisheye camera

## Introduction
This tool is designed for calibrating fisheye cameras and undistorting images taken with them. You can use it to calibrate your camera using a set of images, save the calibration data, and then undistort images using the saved calibration.

## Installation
Installing requirements:
```bash
pip install -r requirements.txt
```


## Usage

### Calibration and Undistortion

To calibrate your camera and undistort images, follow these steps:

1. **Calibration**: Place your calibration images in a directory. These images should be of a calibration pattern viewed from different angles.

2. **Run the Calibration**: Use the following command to calibrate your camera. This will also undistort the images in the input directory and save them to the output directory if the `-s` flag is set to `True`.

```bash
python main.py -i <input_directory> -s <True/False> -c <calibration_file_path> -o <output_directory>
```

- `-i` or `--input_dir`: Directory containing the images to be undistorted. Default is `./images`.
- `-s` or `--save`: Whether to save the calibration results. Default is `False`.
- `-c` or `--calibration_save_path`: Path where the calibration file will be saved. Default is `./calibration.json`.
- `-o` or `--output_dir`: Directory where the undistorted images will be saved. Default is `images_undistorted`.

### Loading Existing Calibration

If you have already calibrated your camera and just want to undistort new images:

1. **Place your images** in a directory.

2. **Run the Tool** with the calibration file path specified:

```bash
python main.py -l <calibration_file_path> -i <input_directory> -o <output_directory>
```

- `-l` or `--calibration_load_path`: Path to the calibration file. If this is provided, the tool will skip calibration and proceed to undistort images.
- `-i` or `--input_dir`: Directory of the images to be undistorted.
- `-o` or `--output_dir`: Directory where the undistorted images will be saved.

For more details about how the callibration process goes, please refer to `callibrator.py`.

## References
- [Calibrate fisheye lens using OpenCV](https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0)