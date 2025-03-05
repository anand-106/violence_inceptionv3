# Violence Detection using InceptionV3

This project implements a violence detection system using the InceptionV3 model to classify video frames as either containing violence or not. The system can be used for real-time detection through a webcam or for analyzing pre-recorded videos.

## Project Structure

```
violence_inceptionv3/
├── data/
│   ├── Violence/            # Place violence videos here
│   ├── NonViolence/         # Place non-violence videos here
│   ├── Violence_frames/     # Extracted frames from violence videos (generated)
│   └── NonViolence_frames/  # Extracted frames from non-violence videos (generated)
├── models/                  # Directory for saved models (generated)
├── extract_frames.py        # Script to extract frames from videos
├── train_model.py           # Script to train the model
├── predict_video.py         # Script to predict on a video file
├── webcam_detection.py      # Script for real-time webcam detection
├── utils.py                 # Utility functions
├── check_gpu.py             # Script to check GPU setup
├── setup_gpu.py             # Script to help set up GPU support
└── requirements.txt         # Python dependencies
```

## Requirements

- Python 3.8+
- TensorFlow 2.16+
- OpenCV
- NumPy
- Matplotlib
- tqdm

For GPU acceleration (recommended):
- NVIDIA GPU
- NVIDIA CUDA Toolkit 12.x
- cuDNN 8.9.x

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/violence_inceptionv3.git
cd violence_inceptionv3
```

2. Install the required packages:
```
pip install -r requirements.txt
```

3. Check GPU setup (optional but recommended):
```
python check_gpu.py
```

4. If GPU setup is needed, run the setup assistant:
```
python setup_gpu.py
```

## Usage

### 1. Prepare Your Dataset

Place your violence videos in the `data/Violence/` directory and non-violence videos in the `data/NonViolence/` directory. Supported formats are `.mp4`, `.avi`, and `.mov`.

### 2. Extract Frames

Extract frames from the videos:
```
python extract_frames.py
```

This will create two directories: `data/Violence_frames/` and `data/NonViolence_frames/` containing the extracted frames.

Options:
- `--violence_dir`: Directory containing violence videos (default: "data/Violence")
- `--non_violence_dir`: Directory containing non-violence videos (default: "data/NonViolence")
- `--output_dir`: Directory to save extracted frames (default: "data")
- `--sample_rate`: Extract one frame every 'sample_rate' frames (default: 30)

### 3. Train the Model

Train the violence detection model:
```
python train_model.py
```

The trained model will be saved to `models/violence_detection_model.keras`.

The script automatically:
- Detects if a GPU is available and configures it
- Limits the number of samples per class to avoid memory issues
- Uses early stopping to prevent overfitting
- Saves the best model based on validation accuracy

### 4. Run Detection

#### Real-time Webcam Detection
```
python webcam_detection.py
```

#### Predict on a Video File
```
python predict_video.py --video path/to/your/video.mp4
```

## Memory Optimization

This project includes memory optimization techniques to handle large datasets:

1. **Sample Limiting**: The `load_dataset` function in `utils.py` limits the number of samples per class to avoid memory issues.
2. **Batch Processing**: Images are processed in batches to reduce memory usage.
3. **Dynamic Batch Size**: The training batch size is adjusted based on GPU availability.

## GPU Setup

For optimal performance, using a GPU is recommended. The project includes two scripts to help with GPU setup:

1. **check_gpu.py**: Checks if your GPU is properly configured for TensorFlow.
2. **setup_gpu.py**: Helps set up CUDA, cuDNN, and TensorFlow with GPU support.

If you encounter the error `numpy.core._exceptions._ArrayMemoryError`, try reducing the `max_samples_per_class` parameter in `train_model.py`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The InceptionV3 model is pre-trained on ImageNet.
- This project was inspired by the need for automated violence detection in video surveillance systems. 