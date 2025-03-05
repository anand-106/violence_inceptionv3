import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import argparse
import os

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Real-time violence detection using webcam")
    parser.add_argument("--model", type=str, default="models/violence_detection_model.h5",
                        help="Path to the trained model")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Violence detection threshold")
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        # Try alternative file extensions
        alternative_paths = [
            args.model.replace('.h5', '.keras'),
            args.model.replace('.h5', '_finetuned.h5')
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                args.model = alt_path
                print(f"Found model at alternative path: {args.model}")
                break
        else:
            print(f"Error: Could not find model file at {args.model} or alternative paths")
            return
    
    # Load the model
    print(f"Loading model from {args.model}...")
    try:
        model = load_model(args.model)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Please make sure you have trained the model using train_model.py")
        return
    
    # Initialize webcam
    print(f"Opening camera device {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera device {args.camera}")
        return
    
    # Get webcam properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera resolution: {frame_width}x{frame_height}, FPS: {fps}")
    
    # Initialize variables for FPS calculation
    fps_counter = 0
    fps_start_time = time.time()
    display_fps = 0
    
    # Initialize prediction smoothing
    prediction_history = []
    history_size = 10  # Number of frames to average
    
    print("Starting detection. Press 'q' to quit.")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame from camera")
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
        result = "Violence" if violence_prob > args.threshold else "No Violence"
        color = (0, 0, 255) if result == "Violence" else (0, 255, 0)
        
        # Display information on frame
        cv2.putText(frame, f"{result}: {violence_prob:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Calculate and display FPS
        fps_counter += 1
        if (time.time() - fps_start_time) > 1:
            display_fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()
        
        cv2.putText(frame, f"FPS: {display_fps:.1f}", (10, frame_height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display the frame
        cv2.imshow("Violence Detection", frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped")

if __name__ == "__main__":
    main() 