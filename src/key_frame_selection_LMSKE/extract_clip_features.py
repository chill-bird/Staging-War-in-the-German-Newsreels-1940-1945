#!/usr/bin/env python3
"""
Extract CLIP image features from a video file.

- Uses OpenCLIP ViT-L/14 (768-dimensional embeddings)
- Samples frames at a fixed FPS
- Encodes frames in batches
- Saves features + metadata as a .pkl file

This script is designed to be called once per video,
e.g. from a Slurm array job.
"""

import argparse
import os
import pickle
import cv2
import torch
import open_clip
import numpy as np
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract CLIP image features from a video"
    )
    parser.add_argument("video_path", type=str, help="Path to input video (.mp4)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="clip_features",
        help="Directory to store .pkl files",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frame sampling rate (frames per second)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for CLIP encoding",
    )
    return parser.parse_args()


def main():
    args = parse_args()

     # Basic paths and output setup
    video_path = args.video_path
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{video_name}.pkl")

    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load OpenCLIP (multilingual-friendly, 768 dims)
    model, preprocess, _ = open_clip.create_model_and_transforms(
        model_name="ViT-L-14",
        pretrained="laion2b_s32b_b82k",
    )
    model = model.to(device)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(int(round(video_fps / args.fps)), 1)

    features = []
    frame_ids = []

    batch = []
    batch_ids = []

    frame_idx = 0
    sampled_idx = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Reading frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Convert OpenCV BGR image → RGB PIL image
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

             # Apply CLIP preprocessing
            batch.append(preprocess(img))
            batch_ids.append(frame_idx)
            sampled_idx += 1

            # If batch is full, run CLIP encoding
            if len(batch) == args.batch_size:
                with torch.no_grad():
                    batch_tensor = torch.stack(batch).to(device)
                    emb = model.encode_image(batch_tensor)
                    emb = emb / emb.norm(dim=-1, keepdim=True)

                    # L2 normalization
                    features.append(emb.cpu().numpy())
                    frame_ids.extend(batch_ids)

                batch.clear()
                batch_ids.clear()

        frame_idx += 1
        pbar.update(1)

    # Flush remaining batch
    if batch:
        with torch.no_grad():
            batch_tensor = torch.stack(batch).to(device)
            emb = model.encode_image(batch_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            features.append(emb.cpu().numpy())
            frame_ids.extend(batch_ids)

    pbar.close()
    cap.release()

    features = np.vstack(features)

    data = {
        "video": video_path,
        "fps": args.fps,
        "frame_indices": frame_ids,
        "features": features,  # shape: [N, 768]
    }

    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Saved features to {output_path}")
    print(f"Feature shape: {features.shape}")


if __name__ == "__main__":
    main()
