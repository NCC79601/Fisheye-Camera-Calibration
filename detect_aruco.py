import cv2
import cv2.aruco as aruco
import numpy as np
from datetime import datetime
from colorama import Fore
import os
import time
import json
from callibrator import Callibrator
from vector_plotter import VectorPlotter

callibrator = Callibrator()
callibrator.load_calibration()

mtx, dist = callibrator.get_pin_hole_intrinsics()

print(f'Pinhole intrinsics:')
print(f'mtx:  {mtx}')
print(f'dist: {dist}')

# load aruco tags config
with open('./aruco_tag_config.json', 'r') as f:
    aruco_config = json.load(f)

tags = aruco_config['tags']
tag_pos  = {}
tag_norm = {}
for tag in tags:
    tag_pos[tag['id']]  = np.array(tag['pos'])
    tag_norm[tag['id']] = np.array(tag['norm'])

print('Initializing camera pose plotter...')
plotter = VectorPlotter()

time.sleep(1)

def main():
    # initialize the camera
    cap = cv2.VideoCapture(1)

    while True:
        # read image
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
        frame = aruco.drawDetectedMarkers(frame, corners, ids)

        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.2, mtx, dist)
        tag_tvec_dict = {}
        tag_rvec_dict = {}

        tags_of_interest = [tag['id'] for tag in tags]
        print(f'tags of interest: {tags_of_interest}')
        print(f'detected ids: {ids}')

        # if markers detected, mark them in the image
        if ids is not None:
            for i, id in enumerate(ids):
                id = id[0]
                if id in tags_of_interest:
                    tag_rvec_dict[id] = rvec[i, 0, (0,1,2)]
                    tag_tvec_dict[id] = tvec[i, 0, (0,1,2)]
                    # test output for tag #2
                    if id == 2:
                        print(f'tag #2: rvec: {rvec[i, 0, :]} tvec: {tvec[i, 0, :]}')
                    frame = cv2.drawFrameAxes(frame, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.03)

        # calculate camera pose
        cam_tvec_list = []
        cam_rvec_list = []

        R_x_90 = cv2.Rodrigues(np.pi/2 * np.array([1, 0, 0]))[0]

        if ids is not None:
            for id in ids:
                id = id[0]
                if id in tags_of_interest:
                    tvec = tag_tvec_dict[id]
                    rvec = tag_rvec_dict[id]
                    R, _ = cv2.Rodrigues(rvec)
                    tvec_r = - R.T @ (tvec)
                    cam_tvec_list.append(tvec_r)
                    cam_rvec_list.append(rvec)
        
        # broute force: average the tvec and rvec
        # TODO: maybe use mean squared error to find the best pose?
        if len(cam_tvec_list):
            cam_pos  = np.mean(cam_tvec_list, axis=0)
            cam_rvec = np.mean(cam_rvec_list, axis=0)
            cam_R = cv2.Rodrigues(cam_rvec)[0]
            cam_R = R_x_90.T @ cam_R
            cam_right = cam_R[0, :]
            cam_front = cam_R[1, :]
            cam_up    = cam_R[2, :]

            print(f'{Fore.GREEN}Camera pose of current frame:{Fore.RESET}')
            print(f' > Camera position:')
            print(f'     pos:   {cam_pos}')
            print(f' > Camera orientation:')
            print(f'     right: {cam_right}')
            print(f'     front: {cam_front}')
            print(f'     up:    {cam_up}')

            plotter.update_vectors(
                [cam_right, cam_front, cam_up],
                cam_pos
            )

        else:
            print(f'{Fore.RED}No tag detected in current frame.{Fore.RESET}')
        
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
        elif key & 0xFF == ord('p'):
            # pause for 5s
            print('Pause for 5 seconds...')
            time.sleep(5)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()