import cv2
import os
import numpy as np
from itertools import groupby
from tqdm import tqdm



# Open the video file
input_folder = './results/utopilot_sun2rain_reduced/test_latest/images/real_B'
output_folder = './results/utopilot_sun2rain_reduced/test_latest/images/real_B_resized'
os.makedirs(output_folder, exist_ok=True)

images = [f for f in os.listdir(input_folder) if f.endswith('.png')]


for img in tqdm(images):
    frame = cv2.imread(f"{input_folder}/{img}")
    frame = cv2.resize(frame, (224, 224))

    cv2.imwrite(f"{output_folder}/{img}", frame)




