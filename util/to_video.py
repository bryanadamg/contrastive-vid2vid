import cv2
import os


# Open the video file
input_folder = './results/utopilot_sun2rain_reduced/test_latest/images/fake_B'
real_folder = './results/utopilot_sun2rain_reduced/test_latest/images/real_A'

output_folder = './results/utopilot_sun2rain_reduced/test_latest'
os.makedirs(output_folder, exist_ok=True)


B_images = os.listdir(input_folder)
A_images = os.listdir(real_folder)


frame_0 = cv2.imread(os.path.join(input_folder, B_images[0]))
height, width, layers = frame_0.shape
video = cv2.VideoWriter(f'{output_folder}/video.avi', 0, 15, (width*2,height))
for frame in sorted(B_images):
    fb = cv2.imread(f"{input_folder}/{frame}")
    fa = cv2.imread(f"{real_folder}/{frame}")
    f = cv2.hconcat([fa, fb])
    video.write(f)

video.release()