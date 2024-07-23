import json
import cv2
import numpy as np
import glob
import os

with open('calibration.json') as f:
    data = json.load(f)
    DIM = data['DIM']
    K = np.array(data['K'])
    D = np.array(data['D'])

def undistort(img_path):
    img = cv2.imread(img_path)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

if __name__ == '__main__':
    image_files = glob.glob('./images/*.png')
    print(f'Number of images to undistort: {len(image_files)}')
    
    if not os.path.exists('./images_undistorted'):
        os.mkdir('./images_undistorted')

    for img_file in image_files:
        undistorted_img = undistort(img_file)
        # save it to ./images_undistorted/
        img_name = img_file.split('/')[-1]
        cv2.imwrite(f'./images_undistorted/{img_name}', undistorted_img)