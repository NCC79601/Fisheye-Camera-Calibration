# reference: https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0

import cv2
import numpy as np
import glob
import json
import os
from typing import Union
from colorama import Fore

class Callibrator(object):
    '''
    Class for calibrating fisheye camera.

    Methods:
    - calibrate: calibrate the camera using specified (save_dir(s) to) images
    - save_calibration: save the calibration results into a json file
    - load_calibration: load the calibration results from a json file
    - undistort: undistort the given images
    '''
    def __init__(
            self,
            checkerboard_shape: tuple = (6, 9),
            subpix_criteria: tuple = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.1
            ),
            calibration_flags: int = \
                cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + \
                cv2.fisheye.CALIB_CHECK_COND + \
                cv2.fisheye.CALIB_FIX_SKEW
        ) -> None:
        '''
        Initialize the callibrator.

        Parameters:
        - checkerboard_shape: tuple, shape of the checkerboard
        - subpix_criteria: tuple, criteria for subpixel accuracy
        - calibration_flags: int, flags for calibration

        See OpenCV documentation for more details.
        '''
        self.CHECKERBOARD      = checkerboard_shape
        self.subpix_criteria   = subpix_criteria
        self.calibration_flags = calibration_flags

        self.objp = np.zeros(
            (1, self.CHECKERBOARD[0] * self.CHECKERBOARD[1], 3),
            np.float32
        )
        self.objp[0, :, :2] = np.mgrid[
            0:self.CHECKERBOARD[0],
            0:self.CHECKERBOARD[1]
        ].T.reshape(-1, 2)

        self.objpoints = []
        self.calibration = None
    
    def calibrate(self, images: Union[list, str]) -> dict:
        '''
        Calibrate the camera using specified (save_dir(s) to) images.

        Parameters:
        - images: can be a list or string as
            - a list of images (MatLike: np.ndarray, cv2.Mat, list, etc)
            - a list of paths to the images (list)
            - a string of save_dir to the folder containing all images (str)
        '''
        images_list = []

        # hander different types of input
        if isinstance(images, str):
            # single save_dir to images
            image_paths = glob.glob('./images/*.png') + glob.glob('./images/*.jpg')
            for img_path in image_paths:
                img = cv2.imread(img_path)
                images_list.append(img)
        elif isinstance(images, list):
            if isinstance(images[0], str):
                # a list of paths to the images
                for img_path in images:
                    img = cv2.imread(img_path)
                    images_list.append(img)
            else:
                # a list of images
                images_list = images
        
        _img_shape = None
        imgpoints = []

        # get image points
        print("Finding chessboard corners...")
        for img in images_list:
            if _img_shape is None:
                _img_shape = img.shape[:2]
            else:
                assert _img_shape == img.shape[:2], "All images must share the same size."
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, self.CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            # If found, add object points, image points (after refining them)
            if ret:
                self.objpoints.append(self.objp)
                cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), self.subpix_criteria)
                imgpoints.append(corners)
        
        N_OK = len(self.objpoints)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

        # calibrate
        print("Running calibration...")
        rms, _, _, _, _ = cv2.fisheye.calibrate(
            self.objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            self.calibration_flags,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )

        print("Found " + str(N_OK) + " valid images for calibration")
        print(Fore.BLUE + "Calibration results:" + Fore.RESET)
        print(Fore.BLUE, end="")
        print("DIM = " + str(_img_shape[::-1]))
        print("K = np.array(" + str(K.tolist()) + ")")
        print("D = np.array(" + str(D.tolist()) + ")")
        print(Fore.RESET, end="")

        # save calibration results
        calibration = {
            'DIM': _img_shape[::-1],
            'K': K.tolist(),
            'D': D.tolist()
        }
        self.calibration = calibration
        return calibration
    
    def save_calibration(self, save_path: str = './calibration.json') -> None:
        '''
        Save the calibration results into a json file

        Parameters:
        - save_path: path for the json file
        '''
        if self.calibration is None:
            raise ValueError('No calibration data found. Please calibrate first.')
        with open(save_path, 'w') as f:
            json.dump(self.calibration, f, indent=4)
        print(f'Results saved in {save_path}')
    
    def load_calibration(self, calibration_path: str = './calibration.json') -> dict:
        '''
        Load the calibration results from a json file.

        Parameters:
        - calibration_path: path to the existing calibration json file

        Returns:
        - calibration: calibration results
        '''
        with open(calibration_path, 'r') as f:
            calibration = json.load(f)
        self.calibration = calibration
        return calibration

    def undistort(
            self,
            images: Union[list, str],
            save: bool = False,
            save_dir = './images_undistorted'
        ):
        '''
        Undistort the image.

        Parameters:
        - images: can be a list or string as
            - a list of images (MatLike: np.ndarray, cv2.Mat, list, etc)
            - a list of paths to the images (list)
            - a string of directory of the images to be distorted (str)
        
        Return:
        - undistorted_imgs: list of undistorted images
        '''
        if self.calibration is None:
            raise ValueError('No calibration data found. Please calibrate first.')
        K = np.array(self.calibration['K'])
        D = np.array(self.calibration['D'])
        DIM = self.calibration['DIM']
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)

        # hander different types of input
        images_list = []
        if isinstance(images, str):
            # single save_dir to images
            image_paths = glob.glob(os.path.join(images, '*.png')) + glob.glob(os.path.join(images, '*.jpg'))
            for img_path in image_paths:
                img = cv2.imread(img_path)
                images_list.append({
                    "img": img,
                    "filename": os.path.basename(img_path)
                })
        elif isinstance(images, list):
            if isinstance(images[0], str):
                # a list of paths to the images
                for img_path in images:
                    img = cv2.imread(img_path)
                    images_list.append({
                        "img": img,
                        "filename": os.path.basename(img_path)
                    })
            else:
                # a list of images
                images_list = [{
                    "img": img,
                    "filename": f'{i}.jpg'
                } for i, img in enumerate(images)]

        # undistort
        undistorted_imgs = []
        for img_dict in images_list:
            img = img_dict["img"]
            undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            undistorted_imgs.append({
                "img": undistorted_img,
                "filename": img_dict["filename"]
            })

        # save the undistorted image
        if save:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            for img_dict in undistorted_imgs:
                img_name = img_dict["filename"]
                undistorted_img = img_dict["img"]
                cv2.imwrite(f'{save_dir}/{img_name}', undistorted_img)
            print(f'Undistorted image saved in {save_dir}')

        return [dict["img"] for dict in undistorted_imgs]