import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from utils import load_dataset, split_dataset, plot_training_history
import tensorflow as tf
from datetime import datetime

def check_gpu():
    """Check if GPU is available and configure it for TensorFlow."""
    print("\n" + "="*50)
    print("CHECKING GPU STATUS")
    print("="*50)
    
    # Check if NVIDIA GPU is available using nvidia-smi
    try:
        nvidia_smi_output = subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT)
        print("NVIDIA GPU detected via nvidia-smi:")
        print(nvidia_smi_output.decode('utf-8').split('\n')[0])
        has_nvidia_gpu = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("No NVIDIA GPU detected via nvidia-smi command.")
        has_nvidia_gpu = False
    
    # Check if TensorFlow can see the GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"TensorFlow detected {len(gpus)} GPU(s):")
            for gpu in gpus:
                print(f"  - {gpu.name}")
            has_tf_gpu = True
        except RuntimeError as e:
            print(f"Error configuring GPU: {e}")
            has_tf_gpu = False
    else:
        print("TensorFlow could not detect any GPUs.")
        has_tf_gpu = False
    
    # Print TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    
    return has_nvidia_gpu and has_tf_gpu

def build_model(input_shape=(224, 224, 3)):
    """Build and compile the InceptionV3 model for binary violence detection."""
    print("Building model for binary violence detection...")
    
    # Load the InceptionV3 model pre-trained on ImageNet
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Add custom layers for binary classification
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)  # Add dropout to prevent overfitting
    predictions = Dense(2, activation='softmax')(x)  # 2 classes: [NonViolence, Violence]
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model, base_model

def train_with_data_generators(data_dir, batch_size=32, epochs=15, img_size=(224, 224)):
    """Train the model using data generators to avoid memory issues."""
    # Create data generators with data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 20% for validation
    )
    
    # Define class mapping explicitly
    class_mapping = {
        'NonViolence_frames': 0,
        'Violence_frames': 1
    }
    
    # Create training generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        classes=list(class_mapping.keys()),
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Create validation generator
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        classes=list(class_mapping.keys()),
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    print(f"Found {train_generator.samples} training samples")
    print(f"Found {validation_generator.samples} validation samples")
    print(f"Class mapping: {train_generator.class_indices}")
    
    # Build the model
    model, base_model = build_model(input_shape=img_size + (3,))
    
    # Print model summary
    model.summary()
    
    # Create directory for model checkpoints
    os.makedirs('models', exist_ok=True)
    
    # Create a timestamp for unique log directory
    log_dir = os.path.join('logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    
    # Define callbacks
    callbacks = [
        # Save the best model based on validation accuracy
        ModelCheckpoint(
            'models/violence_detection_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Stop training when validation accuracy doesn't improve
        EarlyStopping(
            monitor='val_accuracy',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        # TensorBoard logging
        TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]
    
    # Calculate steps per epoch
    steps_per_epoch = train_generator.samples // batch_size
    validation_steps = validation_generator.samples // batch_size
    
    # Ensure at least one step
    steps_per_epoch = max(1, steps_per_epoch)
    validation_steps = max(1, validation_steps)
    
    print(f"Training with {train_generator.samples} samples in {steps_per_epoch} steps per epoch")
    print(f"Validating with {validation_generator.samples} samples in {validation_steps} steps per epoch")
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\nTraining completed!")
    print(f"Model saved to models/violence_detection_model.h5")
    
    # Fine-tune the model by unfreezing some layers
    print("\nFine-tuning the model...")
    
    try:
        # Unfreeze the last 50 layers of the base model
        for layer in base_model.layers[-50:]:
            layer.trainable = True
            
        # Count trainable and non-trainable parameters
        trainable_count = sum(layer.count_params() for layer in base_model.layers if layer.trainable)
        non_trainable_count = sum(layer.count_params() for layer in base_model.layers if not layer.trainable)
        print(f"Trainable parameters in base model: {trainable_count}")
        print(f"Non-trainable parameters in base model: {non_trainable_count}")
        
        # Recompile the model with a lower learning rate
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Update the checkpoint path for fine-tuning
        callbacks[0] = ModelCheckpoint(
            'models/violence_detection_model_finetuned.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        # Fine-tune the model
        history_ft = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=10,  # Fewer epochs for fine-tuning
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nFine-tuning completed!")
        print(f"Fine-tuned model saved to models/violence_detection_model_finetuned.h5")
        
    except Exception as e:
        print(f"\nError during fine-tuning: {e}")
        print("Skipping fine-tuning phase. The initial model is still saved and can be used.")
    
    return model, history

def main():
    # Check GPU status
    has_gpu = check_gpu()
    
    # Set batch size based on GPU availability
    # Smaller batch size for CPU to avoid memory issues
    batch_size = 32 if has_gpu else 8
    print(f"Using batch size: {batch_size}")
    
    # Set data directory
    data_dir = 'data'
    
    # Train the model using data generators
    try:
        model, history = train_with_data_generators(
            data_dir=data_dir,
            batch_size=batch_size,
            epochs=20,
            img_size=(224, 224)
        )
        
    except Exception as e:
        print(f"\nError during model training: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure your dataset directory structure is correct")
        print("2. Ensure you have extracted frames using extract_frames.py")
        print("3. If you're still having memory issues, try reducing the batch size")
        print("4. Make sure your data directory contains class subdirectories with images")
        sys.exit(1)

if __name__ == "__main__":
    main() 