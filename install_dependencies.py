import subprocess
import sys

def install(package):
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    packages = [
        "numpy==1.26.3",
        "opencv-python==4.8.1.78",
        "tensorflow==2.16.1",
        "matplotlib==3.8.2",
        "scikit-learn==1.4.0",
        "pillow==10.2.0",
        "tqdm==4.66.1",
        "imutils==0.5.4"
    ]
    
    for package in packages:
        try:
            install(package)
            print(f"Successfully installed {package}")
        except Exception as e:
            print(f"Error installing {package}: {str(e)}")
    
    print("\nInstallation complete. If there were any errors, try installing those packages manually.")

if __name__ == "__main__":
    main() 