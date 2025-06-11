#!/usr/bin/env python3
"""
Setup script for NL2Bash project.
Helps with dependency installation and environment verification.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, check=True):
    """Run a command and optionally check its return code."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        return False
    return True

def check_python_version():
    """Check if Python version is compatible."""
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 8):
        print(f"❌ Python {major}.{minor} is not supported. Please use Python 3.8+")
        return False
    print(f"✅ Python {major}.{minor} is compatible")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    # Check if pip is available
    if not run_command("pip --version", check=False):
        print("❌ pip is not available. Please install pip first.")
        return False
    
    # Install requirements
    if os.path.exists("requirements.txt"):
        success = run_command("pip install -r requirements.txt")
        if success:
            print("✅ Dependencies installed successfully")
        return success
    else:
        print("❌ requirements.txt not found")
        return False

def verify_installation():
    """Verify that all dependencies are correctly installed."""
    print("\n🔍 Verifying installation...")
    
    required_packages = [
        "torch",
        "transformers", 
        "numpy",
        "pandas",
        "sklearn",
        "tqdm",
        "matplotlib"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        return False
    
    print("\n✅ All packages are available")
    return True

def check_data_file():
    """Check if the data file exists."""
    print("\n📊 Checking data file...")
    
    if os.path.exists("nl2bash-data.json"):
        file_size = os.path.getsize("nl2bash-data.json") / (1024 * 1024)  # MB
        print(f"✅ nl2bash-data.json found ({file_size:.1f} MB)")
        return True
    else:
        print("❌ nl2bash-data.json not found")
        print("Please ensure the data file is in the current directory")
        return False

def test_basic_functionality():
    """Test basic functionality of the modules."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test data preprocessing
        from data_preprocessing import get_data_statistics
        if os.path.exists("nl2bash-data.json"):
            stats = get_data_statistics("nl2bash-data.json")
            print(f"✅ Data preprocessing: {stats['total_samples']} samples found")
        else:
            print("⚠️  Data preprocessing: Skipped (no data file)")
        
        # Test model loading
        from transformer_model import NL2BashModel
        model = NL2BashModel(use_pretrained=True)
        print("✅ Model loading: Successfully created model")
        
        # Test CLI
        if os.path.exists("nl2bash_cli.py"):
            print("✅ CLI interface: Found nl2bash_cli.py")
        else:
            print("❌ CLI interface: nl2bash_cli.py not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = ["checkpoints", "logs"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created/verified directory: {directory}")

def main():
    """Main setup function."""
    print("🚀 NL2Bash Setup Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("\n❌ Setup failed during verification")
        sys.exit(1)
    
    # Check data file
    data_available = check_data_file()
    
    # Test functionality
    if not test_basic_functionality():
        print("\n❌ Setup failed during testing")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Final summary
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\n📚 Next steps:")
    
    if data_available:
        print("1. Test data loading: python3 nl2bash_cli.py test-data")
        print("2. Run demo: python3 demo.py")
        print("3. Start training: python3 nl2bash_cli.py train --num_epochs 5")
        print("4. Use inference: python3 nl2bash_cli.py generate --model_path checkpoints/best_model.pt --interactive")
    else:
        print("1. Add nl2bash-data.json to the current directory")
        print("2. Run setup again: python3 setup.py")
    
    print("\n📖 Read README.md for detailed instructions")

if __name__ == "__main__":
    main() 