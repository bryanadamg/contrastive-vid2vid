import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from scipy.linalg import sqrtm
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import torch.nn.functional as F


from torch.utils.data import DataLoader, Dataset
from PIL import Image


# Load the pretrained I3D model (using ResNet3D here for simplicity)
weights = R3D_18_Weights.KINETICS400_V1
model = r3d_18(weights=weights)
model.eval()
model = nn.Sequential(*list(model.children())[:-1])  # Remove final classification layer

class VideoFramesDataset(Dataset):
    def __init__(self, frames_dir, sequence_length=None, transform=None):
        self.frames_dir = frames_dir
        self.sequence_length = sequence_length
        self.transform = transform

        # Get all frames sorted by filename
        self.all_frames = [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg'))]

        if self.sequence_length != None:
            # Calculate the number of sequences
            self.num_sequences = len(self.all_frames) // self.sequence_length

    def __len__(self):
        if self.sequence_length != None:
            return self.num_sequences
        else:
            return len(self.all_frames)

    def __getitem__(self, idx):
        if self.sequence_length != None:
            start_idx = idx * self.sequence_length
            end_idx = start_idx + self.sequence_length
            sequence_frames = self.all_frames[start_idx:end_idx]

            frames = [Image.open(frame_path) for frame_path in sequence_frames]
            if self.transform:
                frames = [self.transform(frame) for frame in frames]
            frames = torch.stack(frames, dim=0)  # Stack along time dimension (depth)
            return frames
        else:
            frame_path = self.all_frames[idx]
            frame = Image.open(frame_path)
            if self.transform:
                frame = self.transform(frame)
            return frame


# Function to extract features
def extract_features(model, dataloader):
    features = []
    with torch.no_grad():
        for videos in dataloader:
            # The videos tensor shape is now (batch, depth, channels, height, width)
            videos = videos.permute(0, 2, 1, 3, 4)  # Change to (batch, channels, depth, height, width)
            outputs = model(videos)
            features.append(outputs.squeeze(-1).squeeze(-1).squeeze(-1).cpu().numpy())
    features = np.concatenate(features, axis=0)
    return features

# Compute mean and covariance of features
def calculate_stats(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

# Calculate Frechet Distance
def calculate_fvd(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fvd = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fvd

def calculateFVD(real_frames_dir, generated_frames_dir, sequence_length, transform):
    real_dataset = VideoFramesDataset(real_frames_dir, sequence_length, transform=transform)
    generated_dataset = VideoFramesDataset(generated_frames_dir, sequence_length, transform=transform)

    real_loader = DataLoader(real_dataset, batch_size=2, shuffle=False)
    generated_loader = DataLoader(generated_dataset, batch_size=2, shuffle=False)

    real_features = extract_features(model, real_loader)
    generated_features = extract_features(model, generated_loader)

    mu_real, sigma_real = calculate_stats(real_features)
    mu_generated, sigma_generated = calculate_stats(generated_features)

    fvd_score = calculate_fvd(mu_real, sigma_real, mu_generated, sigma_generated)
    print(f"FVD Score: {fvd_score}")

def calculateSSIM_PSNR_MSE(original_frames_dir, generated_frames_dir):
    # Load the datasets
    original_dataset = VideoFramesDataset(original_frames_dir, transform=transform)
    generated_dataset = VideoFramesDataset(generated_frames_dir, transform=transform)

    # Ensure both datasets have the same number of frames
    assert len(original_dataset) == len(generated_dataset), "The number of frames in both directories must be the same."
    # SSIM metric
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
    original_loader = DataLoader(original_dataset, batch_size=1, shuffle=False)
    generated_loader = DataLoader(generated_dataset, batch_size=1, shuffle=False)
    # Calculate SSIM for each corresponding frame
    ssim_scores = []
    with torch.no_grad():
        for (original_frame, generated_frame) in zip(original_loader, generated_loader):
            ssim_score = ssim_metric(original_frame, generated_frame)
            ssim_scores.append(ssim_score.item())

    # Calculate the average SSIM
    average_ssim = sum(ssim_scores) / len(ssim_scores)
    print(f"Average SSIM: {average_ssim}")

    # # Print SSIM for each frame if needed
    # for idx, score in enumerate(ssim_scores):
    #     print(f"Frame {idx + 1}: SSIM = {score}")

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
    # Calculate PSNR for each corresponding frame
    psnr_scores = []
    with torch.no_grad():
        for (original_frame, generated_frame) in zip(original_loader, generated_loader):
            psnr_score = psnr_metric(original_frame, generated_frame)
            psnr_scores.append(psnr_score.item())

    # Calculate the average PSNR
    average_psnr = sum(psnr_scores) / len(psnr_scores)
    print(f"Average PSNR: {average_psnr}")

    # # Print PSNR for each frame if needed
    # for idx, score in enumerate(psnr_scores):
    #     print(f"Frame {idx + 1}: PSNR = {score}")


    # Calculate MSE for each corresponding frame
    mse_scores = []
    with torch.no_grad():
        for (original_frame, generated_frame) in zip(original_loader, generated_loader):
            mse_score = F.mse_loss(original_frame, generated_frame)
            mse_scores.append(mse_score.item())

    # Calculate the average MSE
    average_mse = sum(mse_scores) / len(mse_scores)
    print(f"Average MSE: {average_mse}")

    # # Print MSE for each frame if needed
    # for idx, score in enumerate(mse_scores):
    #     print(f"Frame {idx + 1}: MSE = {score}")



if __name__ == '__main__':

    # Transform for preprocessing frames to keep 224x224 resolution
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    # Example usage
    sequence_length = 16  # Adjust based on your video frame sequence length
    real_frames_dir = './results/fourth_test/test_80/images/real_B'
    generated_frames_dir = './results/fourth_test/test_80/images/fake_B'
    original_frames_dir = './results/fourth_test/test_80/images/real_A'
    real_frames_dir = './results/utopilot_sun2rain_reduced/test_latest/images/real_B'
    generated_frames_dir = './results/utopilot_sun2rain_reduced/test_latest/images/fake_B'
    original_frames_dir = './results/utopilot_sun2rain_reduced/test_latest/images/real_A'

    # calculateFVD(real_frames_dir, generated_frames_dir, sequence_length, transform)
    calculateSSIM_PSNR_MSE(original_frames_dir, generated_frames_dir)
