import os
import sys
import subprocess
import platform
import webbrowser
import ctypes

def is_admin():
    """Check if the script is running with administrator privileges"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def print_section(title):
    """Print a section title with separators"""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80)

def run_command(command, shell=True):
    """Run a command and return its output"""
    try:
        output = subprocess.check_output(command, shell=shell, stderr=subprocess.STDOUT).decode('utf-8')
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

def install_cuda():
    """Guide the user to install CUDA"""
    print_section("CUDA Installation")
    
    print("To install CUDA Toolkit 12.x, follow these steps:")
    print("1. Download CUDA Toolkit from NVIDIA website")
    print("2. Run the installer and follow the instructions")
    print("3. Add CUDA to your PATH environment variable")
    
    open_url = input("Would you like to open the CUDA download page? (y/n): ")
    if open_url.lower() == 'y':
        webbrowser.open("https://developer.nvidia.com/cuda-downloads")
    
    print("\nAfter installation, you need to set the following environment variables:")
    print("CUDA_PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x")
    print("Add to PATH: %CUDA_PATH%\\bin")
    
    set_env = input("Would you like to set these environment variables now? (y/n): ")
    if set_env.lower() == 'y':
        if is_admin():
            # Set CUDA_PATH
            cuda_path = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.7"
            if os.path.exists(cuda_path):
                run_command(f'setx CUDA_PATH "{cuda_path}" /M')
                run_command(f'setx PATH "%PATH%;{cuda_path}\\bin" /M')
                print("[✓] Environment variables set successfully")
            else:
                print("[✗] CUDA path not found. Please install CUDA first.")
        else:
            print("[✗] Administrator privileges required to set environment variables")
            print("Please run this script as administrator")
    
    return True

def install_cudnn():
    """Guide the user to install cuDNN"""
    print_section("cuDNN Installation")
    
    print("To install cuDNN, follow these steps:")
    print("1. Create an NVIDIA Developer account (if you don't have one)")
    print("2. Download cuDNN from NVIDIA website")
    print("3. Extract the files and copy them to your CUDA installation directory")
    
    open_url = input("Would you like to open the cuDNN download page? (y/n): ")
    if open_url.lower() == 'y':
        webbrowser.open("https://developer.nvidia.com/cudnn")
    
    print("\nAfter downloading, follow these steps:")
    print("1. Extract the cuDNN archive")
    print("2. Copy the files to your CUDA installation directory:")
    print("   - Copy bin/* to C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x\\bin")
    print("   - Copy include/* to C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x\\include")
    print("   - Copy lib/* to C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x\\lib")
    
    return True

def install_tensorflow_gpu():
    """Install TensorFlow with GPU support"""
    print_section("TensorFlow GPU Installation")
    
    print("Installing TensorFlow with GPU support...")
    result = run_command(f"{sys.executable} -m pip install tensorflow==2.16.1")
    print(result)
    
    if "Successfully installed" in result:
        print("[✓] TensorFlow installed successfully")
        return True
    else:
        print("[✗] Error installing TensorFlow")
        return False

def main():
    """Main function to set up GPU support"""
    print_section("GPU Setup Assistant")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    
    if not is_admin() and platform.system() == "Windows":
        print("[WARNING] This script is not running with administrator privileges")
        print("Some operations may fail. Consider running as administrator.")
    
    # Check NVIDIA driver
    driver_ok = check_nvidia_driver()
    if not driver_ok:
        print("[✗] NVIDIA driver not detected or not working properly")
        print("Please install the NVIDIA driver first from:")
        print("https://www.nvidia.com/Download/index.aspx")
        open_url = input("Would you like to open the NVIDIA driver download page? (y/n): ")
        if open_url.lower() == 'y':
            webbrowser.open("https://www.nvidia.com/Download/index.aspx")
        return
    
    # Check CUDA
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path and os.path.exists(cuda_path):
        print(f"[✓] CUDA found at {cuda_path}")
    else:
        print("[✗] CUDA not found or not properly configured")
        install_cuda()
    
    # Install cuDNN
    print("\nDo you want to install cuDNN? (Required for TensorFlow GPU support)")
    install_cudnn_choice = input("Install cuDNN? (y/n): ")
    if install_cudnn_choice.lower() == 'y':
        install_cudnn()
    
    # Install TensorFlow with GPU support
    print("\nDo you want to install TensorFlow with GPU support?")
    install_tf_choice = input("Install TensorFlow? (y/n): ")
    if install_tf_choice.lower() == 'y':
        install_tensorflow_gpu()
    
    print_section("Next Steps")
    print("1. Restart your computer to apply all changes")
    print("2. Run 'python check_gpu.py' to verify your GPU setup")
    print("3. If everything is working, run 'python train_model.py' to train your model")

if __name__ == "__main__":
    main() 