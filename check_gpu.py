import subprocess
import sys
import os
import platform
import tensorflow as tf

def print_section(title):
    """Print a section title with separators"""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80)

def run_command(command):
    """Run a command and return its output"""
    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT).decode('utf-8')
        return output
    except subprocess.CalledProcessError as e:
        return f"Error: {e.output.decode('utf-8')}"
    except Exception as e:
        return f"Error: {str(e)}"

def check_nvidia_driver():
    """Check if NVIDIA driver is installed"""
    print_section("NVIDIA Driver Check")
    
    if platform.system() == "Windows":
        output = run_command("wmic path win32_VideoController get name")
        print(output)
        
        if "NVIDIA" in output:
            print("[✓] NVIDIA GPU detected in device list")
            
            driver_output = run_command("nvidia-smi")
            if "NVIDIA-SMI" in driver_output:
                print("[✓] NVIDIA driver is installed and working")
                print(driver_output)
                return True
            else:
                print("[✗] NVIDIA driver is not working properly")
                print("Please download and install the latest NVIDIA driver from:")
                print("https://www.nvidia.com/Download/index.aspx")
                return False
        else:
            print("[✗] No NVIDIA GPU detected")
            return False
    else:
        # Linux/Mac
        output = run_command("lspci | grep -i nvidia")
        if "NVIDIA" in output:
            print("[✓] NVIDIA GPU detected in device list")
            
            driver_output = run_command("nvidia-smi")
            if "NVIDIA-SMI" in driver_output:
                print("[✓] NVIDIA driver is installed and working")
                print(driver_output)
                return True
            else:
                print("[✗] NVIDIA driver is not working properly")
                return False
        else:
            print("[✗] No NVIDIA GPU detected")
            return False

def check_cuda():
    """Check if CUDA is installed and working"""
    print_section("CUDA Check")
    
    if platform.system() == "Windows":
        # Check CUDA_PATH environment variable
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path:
            print(f"[✓] CUDA_PATH environment variable found: {cuda_path}")
            
            # Check nvcc version
            nvcc_output = run_command("nvcc --version")
            if "release" in nvcc_output.lower():
                print("[✓] CUDA compiler (nvcc) is working")
                print(nvcc_output)
            else:
                print("[✗] CUDA compiler (nvcc) is not working")
                print("Make sure CUDA is properly installed and added to PATH")
        else:
            print("[✗] CUDA_PATH environment variable not found")
            print("CUDA might not be installed or not properly configured")
    else:
        # Linux/Mac
        nvcc_output = run_command("nvcc --version")
        if "release" in nvcc_output.lower():
            print("[✓] CUDA compiler (nvcc) is working")
            print(nvcc_output)
        else:
            print("[✗] CUDA compiler (nvcc) is not working")
            print("Make sure CUDA is properly installed and added to PATH")

def check_tensorflow_gpu():
    """Check if TensorFlow can see the GPU"""
    print_section("TensorFlow GPU Check")
    
    print(f"TensorFlow version: {tf.__version__}")
    
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            print(f"[✓] TensorFlow detected {len(physical_devices)} GPU(s):")
            for device in physical_devices:
                print(f"  - {device}")
            
            # Try to allocate memory on GPU
            print("\nTesting GPU memory allocation...")
            try:
                with tf.device('/GPU:0'):
                    a = tf.random.normal([1000, 1000])
                    b = tf.random.normal([1000, 1000])
                    c = tf.matmul(a, b)
                    print(f"[✓] Successfully performed matrix multiplication on GPU")
                    print(f"    Result shape: {c.shape}")
                return True
            except Exception as e:
                print(f"[✗] Error using GPU: {str(e)}")
                return False
        else:
            print("[✗] TensorFlow cannot detect any GPU")
            print("\nPossible reasons:")
            print("1. NVIDIA driver is not installed or outdated")
            print("2. CUDA is not installed or incompatible with TensorFlow")
            print("3. cuDNN is not installed or incompatible")
            print("4. TensorFlow was not installed with GPU support")
            return False
    except Exception as e:
        print(f"[✗] Error checking TensorFlow GPU: {str(e)}")
        return False

def provide_installation_guidance():
    """Provide guidance for installing CUDA and TensorFlow with GPU support"""
    print_section("Installation Guidance")
    
    print("To use TensorFlow with GPU support, you need:")
    print("1. NVIDIA GPU with compute capability 3.5 or higher")
    print("2. NVIDIA driver")
    print("3. CUDA Toolkit")
    print("4. cuDNN SDK")
    print("5. TensorFlow with GPU support")
    
    print("\nFor TensorFlow 2.16.1, you need:")
    print("- CUDA 12.x")
    print("- cuDNN 8.9.x")
    
    print("\nInstallation steps for Windows:")
    print("1. Install NVIDIA driver: https://www.nvidia.com/Download/index.aspx")
    print("2. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
    print("3. Install cuDNN: https://developer.nvidia.com/cudnn (requires NVIDIA account)")
    print("4. Install TensorFlow with GPU support:")
    print("   pip install tensorflow==2.16.1")
    
    print("\nAfter installation, restart your computer and run this script again to verify.")

def main():
    """Main function to check GPU setup"""
    print_section("GPU Setup Check")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    
    driver_ok = check_nvidia_driver()
    check_cuda()
    tf_gpu_ok = check_tensorflow_gpu()
    
    if not driver_ok or not tf_gpu_ok:
        provide_installation_guidance()
    
    print_section("Summary")
    if driver_ok and tf_gpu_ok:
        print("[✓] Your system is properly configured for TensorFlow with GPU support")
        print("You can now train your model with GPU acceleration")
    else:
        print("[✗] Your system is not properly configured for TensorFlow with GPU support")
        print("Please follow the installation guidance above")

if __name__ == "__main__":
    main() 