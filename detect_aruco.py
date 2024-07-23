import cv2
import cv2.aruco as aruco
from callibrator import Callibrator
import numpy as np

callibrator = Callibrator()
callibrator.load_calibration()

def main():
    # initialize the camera
    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture image.")
            break
        frame = np.array(frame)
        frame = callibrator.undistort([frame])[0]
        
        # convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # load aruco dictionary
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters()

        # detect markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # if markers detected, mark them in the image
        if ids is not None:
            frame = aruco.drawDetectedMarkers(frame, corners, ids)

        # display image
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()