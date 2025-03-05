# This script installs TensorFlow compatible with CUDA 11.2
# Run this script as Administrator after setting up CUDA and cuDNN

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "This script needs to be run as Administrator. Please restart it with admin privileges." -ForegroundColor Red
    exit
}

# Configuration
$pythonCmd = "python"

# Print header
Write-Host "TensorFlow for CUDA 11.2 Installation" -ForegroundColor Cyan
Write-Host "------------------------------------" -ForegroundColor Cyan

# Check Python version
Write-Host "`nChecking Python version..." -ForegroundColor Cyan
$pythonVersion = & $pythonCmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Host "Python version: $pythonVersion" -ForegroundColor Green

# Verify pip is installed
Write-Host "`nChecking pip installation..." -ForegroundColor Cyan
try {
    $pipVersion = & $pythonCmd -m pip --version
    Write-Host "pip is installed: $pipVersion" -ForegroundColor Green
} catch {
    Write-Host "pip is not installed. Installing pip..." -ForegroundColor Yellow
    & $pythonCmd -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install pip. Please install pip manually." -ForegroundColor Red
        exit 1
    }
}

# Upgrade pip
Write-Host "`nUpgrading pip to the latest version..." -ForegroundColor Cyan
& $pythonCmd -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to upgrade pip. Continuing anyway..." -ForegroundColor Yellow
} else {
    Write-Host "pip upgraded successfully." -ForegroundColor Green
}

# Uninstall existing TensorFlow installations to avoid conflicts
Write-Host "`nRemoving any existing TensorFlow installations..." -ForegroundColor Cyan
& $pythonCmd -m pip uninstall -y tensorflow tensorflow-gpu
Write-Host "Existing TensorFlow installations removed." -ForegroundColor Green

# Install TensorFlow compatible with CUDA 11.2
Write-Host "`nInstalling TensorFlow 2.10.0 (compatible with CUDA 11.2)..." -ForegroundColor Cyan
& $pythonCmd -m pip install tensorflow==2.10.0
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to install TensorFlow 2.10.0. Please check your internet connection and try again." -ForegroundColor Red
    exit 1
}

# Install compatible numpy version
Write-Host "`nInstalling compatible numpy version..." -ForegroundColor Cyan
& $pythonCmd -m pip install numpy==1.23.5
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to install numpy 1.23.5. This may cause compatibility issues." -ForegroundColor Yellow
}

# Update other dependencies
Write-Host "`nUpdating other dependencies..." -ForegroundColor Cyan
& $pythonCmd -m pip install -r requirements_flexible.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to update some dependencies. This may cause compatibility issues." -ForegroundColor Yellow
}

# Verify TensorFlow installation
Write-Host "`nVerifying TensorFlow installation..." -ForegroundColor Cyan
$tfVersion = & $pythonCmd -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
Write-Host $tfVersion -ForegroundColor Green

# Check GPU detection
Write-Host "`nChecking GPU detection..." -ForegroundColor Cyan
$gpuDetection = & $pythonCmd -c "import tensorflow as tf; print('GPU devices:', tf.config.list_physical_devices('GPU'))"
Write-Host $gpuDetection -ForegroundColor Green

# Provide guidance if GPU is not detected
if ($gpuDetection -like "*[]") {
    Write-Host "`nGPU was not detected by TensorFlow. Please try the following:" -ForegroundColor Yellow
    Write-Host "1. Make sure you've run setup_cuda_env.ps1 as Administrator" -ForegroundColor Yellow
    Write-Host "2. Make sure you've installed cuDNN using install_cudnn.ps1" -ForegroundColor Yellow
    Write-Host "3. Restart your computer to apply environment variable changes" -ForegroundColor Yellow
    Write-Host "4. Run the following command to verify your GPU setup:" -ForegroundColor Yellow
    Write-Host "   python check_gpu.py" -ForegroundColor White
} else {
    Write-Host "`nGPU detected successfully!" -ForegroundColor Green
}

Write-Host "`nTensorFlow installation complete!" -ForegroundColor Green
Write-Host "`nCompatibility Information:" -ForegroundColor Cyan
Write-Host "- TensorFlow 2.10.0 is compatible with CUDA 11.2 and cuDNN 8.1" -ForegroundColor White
Write-Host "- If you haven't installed cuDNN 8.1 yet, please run install_cudnn.ps1" -ForegroundColor White
Write-Host "- You can download cuDNN 8.1 from: https://developer.nvidia.com/cudnn" -ForegroundColor White
Write-Host "  (NVIDIA Developer account required)" -ForegroundColor White 