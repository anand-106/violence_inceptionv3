import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import glob

def extract_frames(video_path, output_dir, sample_rate=30):
    """
    Extract frames from a video file
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save the extracted frames
        sample_rate: Extract one frame every 'sample_rate' frames
    
    Returns:
        count: Number of frames extracted
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return 0
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"Processing {os.path.basename(video_path)}: {frame_count} frames, {fps:.2f} fps, {duration:.2f} seconds")
    
    # Extract frames
    count = 0
    frame_idx = 0
    
    # Use tqdm for progress bar
    with tqdm(total=frame_count, desc=f"Extracting frames from {os.path.basename(video_path)}") as pbar:
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            # Extract one frame every 'sample_rate' frames
            if frame_idx % sample_rate == 0:
                # Save the frame
                frame_path = os.path.join(output_dir, f"{os.path.basename(video_path)}_{count:05d}.jpg")
                cv2.imwrite(frame_path, frame)
                count += 1
            
            frame_idx += 1
            pbar.update(1)
    
    # Release the video
    video.release()
    
    return count

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract frames from videos")
    parser.add_argument("--violence_dir", type=str, default="data/Violence",
                        help="Directory containing violence videos")
    parser.add_argument("--non_violence_dir", type=str, default="data/NonViolence",
                        help="Directory containing non-violence videos")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Directory to save extracted frames")
    parser.add_argument("--sample_rate", type=int, default=30,
                        help="Extract one frame every 'sample_rate' frames")
    args = parser.parse_args()
    
    # Create output directories
    violence_output_dir = os.path.join(args.output_dir, "Violence_frames")
    non_violence_output_dir = os.path.join(args.output_dir, "NonViolence_frames")
    
    os.makedirs(violence_output_dir, exist_ok=True)
    os.makedirs(non_violence_output_dir, exist_ok=True)
    
    # Check if video directories exist
    if not os.path.exists(args.violence_dir):
        print(f"Error: Violence video directory '{args.violence_dir}' not found")
        print("Creating directory. Please add violence videos to this directory.")
        os.makedirs(args.violence_dir, exist_ok=True)
    
    if not os.path.exists(args.non_violence_dir):
        print(f"Error: Non-violence video directory '{args.non_violence_dir}' not found")
        print("Creating directory. Please add non-violence videos to this directory.")
        os.makedirs(args.non_violence_dir, exist_ok=True)
    
    # Get all video files
    violence_videos = glob.glob(os.path.join(args.violence_dir, "*.mp4")) + \
                     glob.glob(os.path.join(args.violence_dir, "*.avi")) + \
                     glob.glob(os.path.join(args.violence_dir, "*.mov"))
    
    non_violence_videos = glob.glob(os.path.join(args.non_violence_dir, "*.mp4")) + \
                         glob.glob(os.path.join(args.non_violence_dir, "*.avi")) + \
                         glob.glob(os.path.join(args.non_violence_dir, "*.mov"))
    
    # Check if videos were found
    if len(violence_videos) == 0:
        print(f"Warning: No video files found in '{args.violence_dir}'")
        print("Please add violence videos (mp4, avi, or mov) to this directory.")
    
    if len(non_violence_videos) == 0:
        print(f"Warning: No video files found in '{args.non_violence_dir}'")
        print("Please add non-violence videos (mp4, avi, or mov) to this directory.")
    
    # Extract frames from violence videos
    print(f"\nProcessing {len(violence_videos)} violence videos...")
    violence_frames_count = 0
    for video_path in violence_videos:
        frames = extract_frames(video_path, violence_output_dir, args.sample_rate)
        violence_frames_count += frames
    
    # Extract frames from non-violence videos
    print(f"\nProcessing {len(non_violence_videos)} non-violence videos...")
    non_violence_frames_count = 0
    for video_path in non_violence_videos:
        frames = extract_frames(video_path, non_violence_output_dir, args.sample_rate)
        non_violence_frames_count += frames
    
    # Print summary
    print("\nExtraction complete!")
    print(f"Violence frames extracted: {violence_frames_count}")
    print(f"Non-violence frames extracted: {non_violence_frames_count}")
    print(f"Total frames extracted: {violence_frames_count + non_violence_frames_count}")
    
    if violence_frames_count == 0 or non_violence_frames_count == 0:
        print("\nWarning: One or both categories have no frames extracted.")
        print("Please ensure you have valid video files in the input directories.")
        print("The model training will fail if one category has no frames.")
    else:
        print("\nYou can now train the model using train_model.py")

if __name__ == "__main__":
    main() 