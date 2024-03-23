import cv2
import os
import numpy as np
from itertools import groupby
from tqdm import tqdm



# Open the video file
input_folder = '/Users/bryanadamg/Documents/Bryan/Documents/SJTU/contrastive-vid2vid/datasets/utopilot_sun2rain/testB'
output_folder = '/Users/bryanadamg/Documents/Bryan/Documents/SJTU/contrastive-vid2vid/datasets/utopilot_sun2rain_downscaled/testB'
os.makedirs(output_folder, exist_ok=True)

images = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]


for img in tqdm(images):
    frame = cv2.imread(f"{input_folder}/{img}")
    frame = cv2.resize(frame, None, fx=0.25, fy=0.25)

    cv2.imwrite(f"{output_folder}/{img}", frame)




