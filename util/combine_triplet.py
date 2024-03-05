import cv2
import os
import numpy as np
from itertools import groupby
from tqdm import tqdm



# Open the video file
input_folder = './sun_day/combined'
output_folder = './trainA'
os.makedirs(output_folder, exist_ok=True)

images = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]

def util_func(x): return x[0:3]

temp = sorted(images, key=util_func)
res = [list(ele) for i, ele in groupby(temp, util_func)]

for video_frame in res:
    frame_count = 0
    sorted_frames = sorted(video_frame)
    for i in tqdm(range(len(sorted_frames)-2)):
        frame_t1 = cv2.imread(f"{input_folder}/{sorted_frames[i]}")
        frame_t2 = cv2.imread(f"{input_folder}/{sorted_frames[i+1]}")
        frame_t3 = cv2.imread(f"{input_folder}/{sorted_frames[i+2]}")

        h, w, c = frame_t1.shape
        x = w/2 - h/2

        frame_t1 = frame_t1[:, int(x):int(x+h)]
        frame_t2 = frame_t2[:, int(x):int(x+h)]
        frame_t3 = frame_t3[:, int(x):int(x+h)]

        concatenated_frame = np.concatenate((frame_t1, frame_t2, frame_t3), axis=1)
        frame_count += 1
        cv2.imwrite(f"{output_folder}/{sorted_frames[i]}", concatenated_frame)


        # cv2.imshow('Concatenated Frames', concatenated_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break


