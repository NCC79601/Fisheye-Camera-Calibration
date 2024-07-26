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


from moveit_comm.client import ClientSocket
client = ClientSocket()


R_x_90 = cv2.Rodrigues(np.pi/2 * np.array([1, 0, 0]))[0]
R_y_90 = cv2.Rodrigues(np.pi/2 * np.array([0, 1, 0]))[0]
R_z_90 = cv2.Rodrigues(np.pi/2 * np.array([0, 0, 1]))[0]
R_xy_180 = cv2.Rodrigues(np.pi * np.array([1, 1, 0]) / np.sqrt(2))[0]
mirror_x = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
tvec_threshold = 5.0

markersX = 4
markersY = 5
markerLength = 35 # mm
markerSeparation = 15  # mm
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.GridBoard((markersX, markersY), float(markerLength), float(markerSeparation), dictionary)

board_norm = np.array([0, 0, 1])
board_pos  = np.array([0, 0, 0])

def main():
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
    tag_pos_dict  = {}
    tag_norm_dict = {}
    for tag in tags:
        tag_pos_dict[tag['id']]  = np.array(tag['pos'])
        tag_norm_dict[tag['id']] = np.array(tag['norm'])

    print('Initializing camera pose plotter...')
    plotter = VectorPlotter()

    time.sleep(1)

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
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters()

        # detect markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        frame = aruco.drawDetectedMarkers(frame, corners, ids)

        if (len(corners) > 0) and (len(ids) > 0):
            print(f'{Fore.GREEN}Detected ids: {ids}{Fore.RESET}')
            rvec = np.zeros((len(ids), 3))
            tvec = np.zeros((len(ids), 3))

            success, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, mtx, dist, rvec, tvec)
            if success > 0:
                # transform the rvec
                R = cv2.Rodrigues(rvec)[0]
                rvec_for_tag_disp = cv2.Rodrigues(R @ R_xy_180)[0]
                print(f' > Found the board with {success} markers.')
                print(f' > rvec: {rvec}')
                print(f' > tvec: {tvec}')
                frame = cv2.drawFrameAxes(frame, mtx, dist, rvec_for_tag_disp, tvec, 100, 3)
            else:
                print(' > Did not find the board.')
                print(f'{Fore.RED}No board detected in current frame.{Fore.RESET}')
        
            # calculate camera pose
            k = np.array([0, 0, 1], dtype=float)
            eps = 1e-6
            if np.linalg.norm(np.cross(k, board_norm)) <= eps:
                rvec_w2tag = np.array([0, 0, 0], dtype=float)
            else:
                theta = np.arccos(np.dot(k, board_norm))
                rvec_w2tag = theta * np.cross(k, board_norm) \
                                / np.linalg.norm(np.cross(k, board_norm)) # checked
            
            R_c2tag = R_xy_180 @ R.T @ R_x_90
            R_w2tag = cv2.Rodrigues(rvec_w2tag)[0]
            R_tag2w = R_w2tag.T
            R_c2w = R_tag2w @ R_c2tag
            R_w2c = R_c2w.T
            
            tvec_tag2c = R_x_90.T @ tvec[:, 0]
            tvec_w2c = - R_c2w @ (tvec_tag2c) + board_pos

            cam_right = R_w2c[0, :]
            cam_front = R_w2c[1, :]
            cam_up    = R_w2c[2, :]

            print(f'{Fore.GREEN}Camera pose of current frame:{Fore.RESET}')
            print(f' > Camera position:')
            print(f'     pos:   {tvec_w2c}')
            print(f' > Camera orientation:')
            print(f'     right: {cam_right}')
            print(f'     front: {cam_front}')
            print(f'     up:    {cam_up}')

            plotter.update_vectors(
                [cam_right, cam_front, cam_up],
                tvec_w2c,
                axis_lim=[-1000, 1000],
                scale=300.0
            )

            rvec_w2c = cv2.Rodrigues(R_w2c)[0].astype(np.float64).flatten()

            pose = np.array([
                tvec_w2c[0],
                tvec_w2c[1],
                tvec_w2c[2],
                rvec_w2c[0],
                rvec_w2c[1],
                rvec_w2c[2],
            ], dtype=np.float64)
            print(f'{Fore.YELLOW}Sending pose to server...{Fore.RESET}')
            print(f' > pose: {pose}')
            client.cli_send(pose)
        
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