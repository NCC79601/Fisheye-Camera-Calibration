import cv2
import cv2.aruco as aruco
from callibrator import Callibrator
import numpy as np
from datetime import datetime
import os
from colorama import Fore


callibrator = Callibrator()
callibrator.load_calibration()

K, D = callibrator.get_intrinsics()
mtx, dist = callibrator.get_pin_hole_intrinsics()

print(f'Pinhole intrinsics:')
print(f'mtx:  {mtx}')
print(f'dist: {dist}')


def main():
    # initialize the camera
    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture image.")
            break
        
        # undistort the image
        frame = callibrator.undistort([frame])[0]
        undrawn_frame = frame.copy()
        
        # convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # load aruco dictionary
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters()


        # detect markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.2, mtx, dist)

        # if markers detected, mark them in the image
        if ids is not None:
            frame = aruco.drawDetectedMarkers(frame, corners, ids)
            for i in range(rvec.shape[0]):
                frame = cv2.drawFrameAxes(frame, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.03)

        # display image
        cv2.imshow('Frame', frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('s'):
            if not os.path.exists('./capture'):
                os.makedirs('./capture')
            now = datetime.now()
            timestamp = now.strftime('%Y-%m-%d-%H:%M:%S.%f')
            filename = f"./capture/image_{timestamp}.jpg"
            print(f'Saved current frame to {Fore.BLUE + filename + Fore.RESET}')
            cv2.imwrite(filename, undrawn_frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()