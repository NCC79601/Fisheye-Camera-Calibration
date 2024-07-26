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

R_x_90 = cv2.Rodrigues(np.pi/2 * np.array([1, 0, 0]))[0]
mirror_x = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
tvec_threshold = 5.0

def main():
    callibrator = Callibrator()
    callibrator.load_calibration()

    mtx, dist = callibrator.get_pin_hole_intrinsics()

    print(f'Pinhole intrinsics:')
    print(f'mtx:  {mtx}')
    print(f'dist: {dist}')

    # load aruco tags config
    aruco_config_path = os.path.join(os.path.dirname(__file__), 'aruco_tag_config.json')
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
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters()

        # detect markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        frame = aruco.drawDetectedMarkers(frame, corners, ids)

        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.2, mtx, dist)
        tag_pose_dict = {}

        tags_of_interest = [tag['id'] for tag in tags]
        print(f'detected ids: {ids}')

        # if markers detected, mark them in the image
        if ids is not None:
            for i, id in enumerate(ids):
                id = id[0]
                if id in tags_of_interest:
                    tag_pose_dict[id] = {
                        "tvec": tvec[i, 0, (0,1,2)],
                        "rvec": rvec[i, 0, (0,1,2)],
                        "tag_pos":  tag_pos_dict[id],
                        "tag_norm": tag_norm_dict[id]
                    }
                    # test output for tag #2
                    frame = cv2.drawFrameAxes(frame, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.03)

        # calculate camera pose
        tvec_w2c_list = []
        rvec_c2w_list = []

        if ids is not None:
            for id in ids:
                id = id[0]
                if id in tags_of_interest:
                    print(f'{Fore.GREEN}Tag {id} detected in current frame:{Fore.RESET}')
                    pose = tag_pose_dict[id]
                    tvec = pose['tvec'] # from camera to tag (in cam coord)
                    rvec = pose['rvec']

                    tag_pos  = pose['tag_pos']
                    tag_norm = pose['tag_norm']
                    tag_norm = tag_norm / np.linalg.norm(tag_norm)

                    k = np.array([0, 0, 1], dtype=float)
                    eps = 1e-6
                    if np.linalg.norm(np.cross(k, tag_norm)) <= eps:
                        rvec_w2tag = np.array([0, 0, 0], dtype=float)
                    else:
                        theta = np.arccos(np.dot(k, tag_norm))
                        rvec_w2tag = theta * np.cross(k, tag_norm) \
                                     / np.linalg.norm(np.cross(k, tag_norm))
                    R_c2tag = cv2.Rodrigues(rvec)[0] @ R_x_90.T # checked
                    R_tag2c = R_c2tag.T # checked
                    R_w2tag = cv2.Rodrigues(rvec_w2tag)[0] # checked
                    R_w2c = (R_tag2c @ R_w2tag)
                    R_c2w = R_w2c.T
                    rvec_c2w = cv2.Rodrigues(R_c2w)[0]
                    
                    tvec_tag2c = R_x_90.T @ tvec
                    tvec_w2c = - R_c2w.T @ (tvec_tag2c) + tag_pos

                    if np.linalg.norm(tvec_w2c) < tvec_threshold:
                        tvec_w2c_list.append(tvec_w2c)
                        rvec_c2w_list.append(rvec_c2w)
                    else:
                        print(f'{Fore.BLUE}Ignore tag {id}.{Fore.RESET}')
        
        # broute force: average the tvec and rvec
        # TODO: maybe use mean squared error to find the best pose?
        if len(tvec_w2c_list):
            tvec_w2c_avg  = np.mean(tvec_w2c_list, axis=0)
            rvec_w2c_avg = np.mean(rvec_c2w_list, axis=0)
            R_w2c = cv2.Rodrigues(rvec_w2c_avg)[0]
            cam_right = R_w2c[0, :]
            cam_front = R_w2c[1, :]
            cam_up    = R_w2c[2, :]

            print(f'{Fore.GREEN}Camera pose of current frame:{Fore.RESET}')
            print(f' > Camera position:')
            print(f'     pos:   {tvec_w2c_avg}')
            print(f' > Camera orientation:')
            print(f'     right: {cam_right}')
            print(f'     front: {cam_front}')
            print(f'     up:    {cam_up}')

            plotter.update_vectors(
                [cam_right, cam_front, cam_up],
                tvec_w2c_avg
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