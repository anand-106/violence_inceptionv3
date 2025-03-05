import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse
import os
import time
from tqdm import tqdm

def predict_video(video_path, model_path, output_path=None, threshold=0.7):
    """
    Process a video file and detect violence in each frame.
    
    Args:
        video_path: Path to the input video file
        model_path: Path to the trained model
        output_path: Path to save the output video (optional)
        threshold: Threshold for violence detection
    """
    # Check if model file exists
    if not os.path.exists(model_path):
        # Try alternative file extensions
        alternative_paths = [
            model_path.replace('.h5', '.keras'),
            model_path.replace('.h5', '_finetuned.h5'),
            model_path.replace('.keras', '.h5')
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                model_path = alt_path
                print(f"Found model at alternative path: {model_path}")
                break
        else:
            print(f"Error: Could not find model file at {model_path} or alternative paths")
            return
    
    # Load the model
    print(f"Loading model from {model_path}...")
    try:
        model = load_model(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Please make sure you have trained the model using train_model.py")
        return
    
    # Open the video file
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")
    
    # Create output video writer if output path is specified
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        print(f"Output video will be saved to: {output_path}")
    
    # Initialize prediction smoothing
    prediction_history = []
    history_size = 10  # Number of frames to average
    
    # Initialize violence detection statistics
    violence_frames = 0
    total_processed_frames = 0
    
    # Process the video
    print("Processing video...")
    pbar = tqdm(total=total_frames)
    
    while True:
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Preprocess the frame
        resized_frame = cv2.resize(frame, (224, 224))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        preprocessed_frame = rgb_frame.astype("float32") / 255.0  # Normalize to [0,1]
        preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)
        
        # Make prediction
        prediction = model.predict(preprocessed_frame, verbose=0)[0]
        
        # Add prediction to history for smoothing
        prediction_history.append(prediction)
        if len(prediction_history) > history_size:
            prediction_history.pop(0)
        
        # Calculate average prediction
        avg_prediction = np.mean(prediction_history, axis=0)
        
        # Get violence probability (second class)
        violence_prob = avg_prediction[1]  # [NonViolence, Violence]
        result = "Violence" if violence_prob > threshold else "No Violence"
        
        # Update statistics
        if result == "Violence":
            violence_frames += 1
        
        # Add visualization to frame
        color = (0, 0, 255) if result == "Violence" else (0, 255, 0)
        cv2.putText(frame, f"{result}: {violence_prob:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Write frame to output video if specified
        if output_path:
            out.write(frame)
        
        total_processed_frames += 1
        pbar.update(1)
    
    # Close resources
    cap.release()
    if output_path:
        out.release()
    pbar.close()
    
    # Calculate and display statistics
    violence_percentage = (violence_frames / total_processed_frames) * 100 if total_processed_frames > 0 else 0
    print(f"\nProcessing complete!")
    print(f"Total frames processed: {total_processed_frames}")
    print(f"Frames with violence detected: {violence_frames} ({violence_percentage:.2f}%)")
    
    if output_path:
        print(f"Output video saved to: {output_path}")
    
    return violence_percentage

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Violence detection in video")
    parser.add_argument("--video", type=str, required=True,
                        help="Path to the input video file")
    parser.add_argument("--model", type=str, default="models/violence_detection_model.h5",
                        help="Path to the trained model")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save the output video (optional)")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Violence detection threshold")
    args = parser.parse_args()
    
    # Process the video
    predict_video(args.video, args.model, args.output, args.threshold)

if __name__ == "__main__":
    main() 