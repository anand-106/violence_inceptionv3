"""
TensorFlow GPU Diagnostic and Fix Tool
-------------------------------------
This script helps diagnose and fix common issues with TensorFlow GPU support.
It checks your CUDA, cuDNN, and TensorFlow setup and provides guidance on fixing issues.
"""

import os
import sys
import subprocess
import platform
import ctypes
import shutil
from pathlib import Path

def is_admin():
    """Check if the script is running with administrator privileges."""
    try:
        if platform.system() == 'Windows':
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        else:
            return os.geteuid() == 0
    except:
        return False

def print_section(title):
    """Print a formatted section title."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def run_command(command, shell=True):
    """Run a command and return its output."""
    try:
        result = subprocess.run(
            command, 
            shell=shell, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"
    except Exception as e:
        return f"Error: {str(e)}"

def check_python_version():
    """Check the Python version."""
    print_section("Python Version")
    version = platform.python_version()
    print(f"Python version: {version}")
    
    # Check if Python version is compatible with TensorFlow 2.10
    major, minor, _ = map(int, version.split('.'))
    if (major == 3 and 7 <= minor <= 9):
        print("✅ Python version is compatible with TensorFlow 2.10")
    else:
        print("⚠️ Python version may not be compatible with TensorFlow 2.10")
        print("   Recommended: Python 3.7-3.9")

def check_tensorflow_version():
    """Check the TensorFlow version."""
    print_section("TensorFlow Version")
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        
        # Check if TensorFlow version is compatible with CUDA 11.2
        if tf.__version__.startswith(('2.10.', '2.9.', '2.8.')):
            print("✅ TensorFlow version is compatible with CUDA 11.2")
        else:
            print("⚠️ TensorFlow version may not be compatible with CUDA 11.2")
            print("   Recommended: TensorFlow 2.10.0 for CUDA 11.2")
    except ImportError:
        print("❌ TensorFlow is not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking TensorFlow: {str(e)}")
        return False
    return True

def check_cuda():
    """Check CUDA installation."""
    print_section("CUDA Installation")
    
    # Check CUDA_PATH environment variable
    cuda_path = os.environ.get('CUDA_PATH')
    if cuda_path:
        print(f"CUDA_PATH: {cuda_path}")
        if os.path.exists(cuda_path):
            print(f"✅ CUDA directory exists: {cuda_path}")
        else:
            print(f"❌ CUDA directory does not exist: {cuda_path}")
    else:
        print("❌ CUDA_PATH environment variable not set")
    
    # Check nvcc version
    nvcc_output = run_command("nvcc --version")
    if "not recognized" in nvcc_output or "Error" in nvcc_output:
        print("❌ nvcc (CUDA compiler) not found in PATH")
    else:
        print("CUDA compiler version:")
        for line in nvcc_output.split('\n'):
            if "release" in line.lower():
                print(f"  {line.strip()}")
                if "11.2" in line:
                    print("✅ CUDA 11.2 detected (compatible with TensorFlow 2.10)")
                else:
                    print("⚠️ CUDA version may not be compatible with TensorFlow 2.10")
                    print("   Recommended: CUDA 11.2 for TensorFlow 2.10")
    
    # Check for CUDA DLLs in PATH
    cuda_dlls = ["cudart64_11.dll", "cublas64_11.dll", "cufft64_10.dll", 
                "curand64_10.dll", "cusolver64_11.dll", "cusparse64_11.dll"]
    
    print("\nChecking for CUDA DLLs in PATH:")
    path_dirs = os.environ.get('PATH', '').split(os.pathsep)
    
    for dll in cuda_dlls:
        found = False
        for path_dir in path_dirs:
            dll_path = os.path.join(path_dir, dll)
            if os.path.exists(dll_path):
                print(f"✅ Found {dll} in {path_dir}")
                found = True
                break
        
        if not found:
            print(f"❌ {dll} not found in PATH")

def check_cudnn():
    """Check cuDNN installation."""
    print_section("cuDNN Installation")
    
    # Check CUDNN_PATH environment variable
    cudnn_path = os.environ.get('CUDNN_PATH')
    if cudnn_path:
        print(f"CUDNN_PATH: {cudnn_path}")
        if os.path.exists(cudnn_path):
            print(f"✅ cuDNN directory exists: {cudnn_path}")
            
            # Check for cuDNN files
            bin_dir = os.path.join(cudnn_path, 'bin')
            include_dir = os.path.join(cudnn_path, 'include')
            lib_dir = os.path.join(cudnn_path, 'lib')
            
            if os.path.exists(bin_dir) and any(f.endswith('.dll') for f in os.listdir(bin_dir)):
                print(f"✅ cuDNN DLLs found in {bin_dir}")
            else:
                print(f"❌ No cuDNN DLLs found in {bin_dir}")
            
            if os.path.exists(include_dir) and any(f.endswith('.h') for f in os.listdir(include_dir)):
                print(f"✅ cuDNN header files found in {include_dir}")
            else:
                print(f"❌ No cuDNN header files found in {include_dir}")
            
            if os.path.exists(lib_dir) and any(f.endswith('.lib') for f in os.listdir(lib_dir)):
                print(f"✅ cuDNN library files found in {lib_dir}")
            else:
                print(f"❌ No cuDNN library files found in {lib_dir}")
        else:
            print(f"❌ cuDNN directory does not exist: {cudnn_path}")
    else:
        print("❌ CUDNN_PATH environment variable not set")
        
        # Try to find cuDNN in CUDA directory
        cuda_path = os.environ.get('CUDA_PATH')
        if cuda_path:
            cudnn_header = os.path.join(cuda_path, 'include', 'cudnn.h')
            if os.path.exists(cudnn_header):
                print(f"✅ Found cuDNN header at {cudnn_header}")
            else:
                print(f"❌ cuDNN header not found in CUDA include directory")

def check_nvidia_driver():
    """Check NVIDIA driver installation."""
    print_section("NVIDIA Driver")
    
    nvidia_smi_output = run_command("nvidia-smi")
    if "not recognized" in nvidia_smi_output or "Error" in nvidia_smi_output:
        print("❌ nvidia-smi not found. NVIDIA driver may not be installed.")
        return False
    else:
        print("NVIDIA driver information:")
        lines = nvidia_smi_output.split('\n')
        for i, line in enumerate(lines):
            if i < 3:  # Print first 3 lines which contain driver version
                print(line)
        return True

def check_tensorflow_gpu():
    """Check if TensorFlow can detect the GPU."""
    print_section("TensorFlow GPU Detection")
    
    try:
        import tensorflow as tf
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            print(f"✅ TensorFlow detected {len(physical_devices)} GPU(s):")
            for device in physical_devices:
                print(f"  - {device.name}")
            
            # Try to perform a simple operation on GPU
            print("\nTesting GPU with a simple matrix multiplication...")
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
                c = tf.matmul(a, b)
                print(f"Result: \n{c.numpy()}")
                print("✅ GPU computation successful")
            return True
        else:
            print("❌ TensorFlow could not detect any GPUs")
            return False
    except ImportError:
        print("❌ TensorFlow is not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking TensorFlow GPU: {str(e)}")
        return False

def fix_cuda_path():
    """Fix CUDA PATH issues."""
    print_section("Fixing CUDA PATH")
    
    if not is_admin():
        print("❌ Administrator privileges required to fix PATH issues")
        print("   Please run this script as Administrator")
        return False
    
    cuda_path = os.environ.get('CUDA_PATH')
    if not cuda_path:
        print("❌ CUDA_PATH environment variable not set")
        return False
    
    # Add CUDA bin directory to PATH
    cuda_bin = os.path.join(cuda_path, 'bin')
    if os.path.exists(cuda_bin):
        path_env = os.environ.get('PATH', '')
        if cuda_bin not in path_env:
            # Update PATH for current process
            os.environ['PATH'] = cuda_bin + os.pathsep + path_env
            print(f"✅ Added {cuda_bin} to PATH for current process")
            
            # Update PATH permanently
            try:
                if platform.system() == 'Windows':
                    # Get current PATH from registry
                    import winreg
                    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 0, winreg.KEY_ALL_ACCESS)
                    current_path, _ = winreg.QueryValueEx(key, 'Path')
                    
                    # Add CUDA bin if not already in PATH
                    if cuda_bin not in current_path:
                        new_path = cuda_bin + os.pathsep + current_path
                        winreg.SetValueEx(key, 'Path', 0, winreg.REG_EXPAND_SZ, new_path)
                        winreg.CloseKey(key)
                        print(f"✅ Added {cuda_bin} to system PATH permanently")
                    else:
                        print(f"✅ {cuda_bin} is already in system PATH")
                else:
                    print("⚠️ Permanent PATH modification not supported on this platform")
            except Exception as e:
                print(f"❌ Error updating system PATH: {str(e)}")
        else:
            print(f"✅ {cuda_bin} is already in PATH")
    else:
        print(f"❌ CUDA bin directory not found: {cuda_bin}")
    
    return True

def fix_cudnn_path():
    """Fix cuDNN PATH issues."""
    print_section("Fixing cuDNN PATH")
    
    if not is_admin():
        print("❌ Administrator privileges required to fix PATH issues")
        print("   Please run this script as Administrator")
        return False
    
    cudnn_path = os.environ.get('CUDNN_PATH')
    if not cudnn_path:
        print("❌ CUDNN_PATH environment variable not set")
        print("   Please run install_cudnn.ps1 to set up cuDNN")
        return False
    
    # Add cuDNN bin directory to PATH
    cudnn_bin = os.path.join(cudnn_path, 'bin')
    if os.path.exists(cudnn_bin):
        path_env = os.environ.get('PATH', '')
        if cudnn_bin not in path_env:
            # Update PATH for current process
            os.environ['PATH'] = cudnn_bin + os.pathsep + path_env
            print(f"✅ Added {cudnn_bin} to PATH for current process")
            
            # Update PATH permanently
            try:
                if platform.system() == 'Windows':
                    # Get current PATH from registry
                    import winreg
                    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 0, winreg.KEY_ALL_ACCESS)
                    current_path, _ = winreg.QueryValueEx(key, 'Path')
                    
                    # Add cuDNN bin if not already in PATH
                    if cudnn_bin not in current_path:
                        new_path = cudnn_bin + os.pathsep + current_path
                        winreg.SetValueEx(key, 'Path', 0, winreg.REG_EXPAND_SZ, new_path)
                        winreg.CloseKey(key)
                        print(f"✅ Added {cudnn_bin} to system PATH permanently")
                    else:
                        print(f"✅ {cudnn_bin} is already in system PATH")
                else:
                    print("⚠️ Permanent PATH modification not supported on this platform")
            except Exception as e:
                print(f"❌ Error updating system PATH: {str(e)}")
        else:
            print(f"✅ {cudnn_bin} is already in PATH")
    else:
        print(f"❌ cuDNN bin directory not found: {cudnn_bin}")
    
    return True

def install_correct_tensorflow():
    """Install the correct version of TensorFlow for CUDA 11.2."""
    print_section("Installing Correct TensorFlow Version")
    
    # Uninstall existing TensorFlow
    print("Uninstalling existing TensorFlow...")
    run_command("pip uninstall -y tensorflow tensorflow-gpu")
    
    # Install TensorFlow 2.10.0
    print("Installing TensorFlow 2.10.0 (compatible with CUDA 11.2)...")
    result = run_command("pip install tensorflow==2.10.0")
    
    if "Successfully installed" in result:
        print("✅ TensorFlow 2.10.0 installed successfully")
        return True
    else:
        print(f"❌ Error installing TensorFlow: {result}")
        return False

def provide_summary(issues):
    """Provide a summary of issues and fixes."""
    print_section("Summary")
    
    if not issues:
        print("✅ No issues detected! Your TensorFlow GPU setup should be working correctly.")
        return
    
    print("The following issues were detected:")
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")
    
    print("\nRecommended actions:")
    print("1. Run setup_cuda_env.ps1 as Administrator to set up environment variables")
    print("2. Run install_cudnn.ps1 as Administrator to install cuDNN")
    print("3. Run install_tensorflow_cuda11.ps1 as Administrator to install the correct TensorFlow version")
    print("4. Restart your computer to apply all changes")
    print("5. Run this script again to verify the fixes")

def main():
    """Main function to diagnose and fix TensorFlow GPU issues."""
    print("TensorFlow GPU Diagnostic and Fix Tool")
    print("-------------------------------------")
    print("This script will diagnose and fix common issues with TensorFlow GPU support.")
    
    issues = []
    
    # Check Python version
    check_python_version()
    
    # Check NVIDIA driver
    if not check_nvidia_driver():
        issues.append("NVIDIA driver not installed or not working properly")
    
    # Check CUDA installation
    check_cuda()
    if not os.environ.get('CUDA_PATH'):
        issues.append("CUDA_PATH environment variable not set")
    
    # Check cuDNN installation
    check_cudnn()
    if not os.environ.get('CUDNN_PATH'):
        issues.append("CUDNN_PATH environment variable not set")
    
    # Check TensorFlow version
    if not check_tensorflow_version():
        issues.append("TensorFlow not installed or incorrect version")
    
    # Check TensorFlow GPU detection
    if not check_tensorflow_gpu():
        issues.append("TensorFlow cannot detect GPU")
    
    # Provide summary and fix options
    provide_summary(issues)
    
    if issues and is_admin():
        print("\nWould you like to attempt to fix these issues? (y/n)")
        choice = input().lower()
        if choice == 'y':
            # Fix CUDA PATH
            fix_cuda_path()
            
            # Fix cuDNN PATH
            fix_cudnn_path()
            
            # Install correct TensorFlow version
            install_correct_tensorflow()
            
            print("\nFixes applied. Please restart your computer and run this script again to verify.")
    elif issues:
        print("\nPlease run this script as Administrator to fix these issues.")

if __name__ == "__main__":
    main() 