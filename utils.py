import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
import glob
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
import random
from sklearn.model_selection import train_test_split

def extract_frames_from_videos(violence_path, non_violence_path):
    """
    Extract frames from videos and save them as images
    
    Args:
        violence_path: Path to violent videos
        non_violence_path: Path to non-violent videos
    """
    # Create directories if they don't exist
    os.makedirs('./data/Violence_frames', exist_ok=True)
    os.makedirs('./data/NonViolence_frames', exist_ok=True)
    
    # Process violence videos
    print("Processing violence videos...")
    for path in tqdm(glob.glob(violence_path + '/*')):
        fname = os.path.basename(path).split('.')[0]
        vidcap = cv2.VideoCapture(path)
        success, image = vidcap.read()
        count = 0
        while success:
            if count % 5 == 0:
                cv2.imwrite("./data/Violence_frames/{}-{}.jpg".format(fname, str(count).zfill(4)), image)
            success, image = vidcap.read()
            count += 1
    
    # Process non-violence videos
    print("Processing non-violence videos...")
    for path in tqdm(glob.glob(non_violence_path + '/*')):
        fname = os.path.basename(path).split('.')[0]
        vidcap = cv2.VideoCapture(path)
        success, image = vidcap.read()
        count = 0
        while success:
            if count % 5 == 0:
                cv2.imwrite("./data/NonViolence_frames/{}-{}.jpg".format(fname, str(count).zfill(4)), image)
            success, image = vidcap.read()
            count += 1

def load_dataset(data_dir, img_size=(224, 224), validation_split=0.2, max_samples_per_class=5000, verbose=True):
    """
    Load and preprocess the dataset.
    
    Args:
        data_dir: Directory containing the dataset
        img_size: Size to resize images to
        validation_split: Fraction of data to use for validation
        max_samples_per_class: Maximum number of samples to load per class
        verbose: Whether to print progress information
    
    Returns:
        X_train: Training images
        y_train: Training labels (one-hot encoded)
        X_val: Validation images
        y_val: Validation labels (one-hot encoded)
    """
    # Print debug information
    if verbose:
        print(f"Loading dataset from: {data_dir}")
        print(f"Max samples per class: {max_samples_per_class}")
    
    # Check if the data directory exists
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")
    
    # Define class labels
    # Look for subdirectories in the data directory
    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if verbose:
        print(f"Found {len(class_dirs)} class directories: {class_dirs}")
    
    if len(class_dirs) == 0:
        raise ValueError(f"No class directories found in {data_dir}. Make sure you have extracted frames using extract_frames.py")
    
    # Initialize lists to store images and labels
    images = []
    labels = []
    
    # Load images from each class
    for class_idx, class_name in enumerate(class_dirs):
        class_dir = os.path.join(data_dir, class_name)
        
        if verbose:
            print(f"Loading class {class_idx}: {class_name} from {class_dir}")
        
        # Get all image files in the class directory
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(image_files) == 0:
            print(f"Warning: No images found in {class_dir}")
            continue
        
        if verbose:
            print(f"  Found {len(image_files)} images")
        
        # Limit the number of samples per class if needed
        if max_samples_per_class and len(image_files) > max_samples_per_class:
            if verbose:
                print(f"  Limiting to {max_samples_per_class} samples")
            # Randomly sample to avoid bias
            image_files = random.sample(image_files, max_samples_per_class)
        
        # Load and preprocess images
        for img_file in tqdm(image_files, desc=f"Loading {class_name}", disable=not verbose):
            img_path = os.path.join(class_dir, img_file)
            try:
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not load image {img_path}")
                    continue
                
                # Resize image
                img = cv2.resize(img, img_size)
                
                # Convert BGR to RGB (OpenCV loads as BGR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Normalize pixel values to [0, 1]
                img = img.astype("float32") / 255.0
                
                # Add to lists
                images.append(img)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    if len(images) == 0:
        raise ValueError("No valid images found in the dataset")
    
    # Convert lists to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    if verbose:
        print(f"Dataset loaded: {X.shape[0]} images, {len(np.unique(y))} classes")
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, stratify=y, random_state=42
    )
    
    # Convert labels to one-hot encoding
    num_classes = len(np.unique(y))
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    
    if verbose:
        print(f"Training set: {X_train.shape[0]} images")
        print(f"Validation set: {X_val.shape[0]} images")
        print(f"Number of classes: {num_classes}")
    
    return X_train, y_train, X_val, y_val

def split_dataset(X, y, test_size=0.2, validation_size=0.1):
    """
    Split dataset into training, validation, and test sets.
    
    Args:
        X: Image data
        y: Labels
        test_size: Proportion of the dataset to include in the test split
        validation_size: Proportion of the training data to include in the validation split
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Calculate sizes
    n_samples = len(X)
    n_test = int(test_size * n_samples)
    n_val = int(validation_size * (n_samples - n_test))
    n_train = n_samples - n_test - n_val
    
    # Create indices
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    # Split the data
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def plot_training_history(H, epochs, output_path="plot.png"):
    """
    Plot the training loss and accuracy
    
    Args:
        H: History object or dictionary containing training history
        epochs: Number of epochs
        output_path: Path to save the plot
    """
    # Create figure
    plt.figure(figsize=(12, 5))
    
    # Get the number of epochs
    N = epochs
    
    # Plot training loss and accuracy
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, N), H["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H["val_loss"], label="val_loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, N), H["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H["val_accuracy"], label="val_acc")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Training history plot saved to {output_path}")

def process_frame_for_prediction(frame):
    """
    Process a single frame for prediction
    
    Args:
        frame: Input frame
        
    Returns:
        Processed frame ready for prediction
    """
    # Define the ImageNet mean subtraction values
    mean = np.array([123.68, 116.779, 103.939], dtype="float32")
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize the frame
    resized_frame = cv2.resize(rgb_frame, (224, 224)).astype("float32")
    
    # Perform mean subtraction
    resized_frame -= mean
    
    # Expand dimensions to match model input shape
    img_array = np.expand_dims(resized_frame, axis=0)
    
    return img_array