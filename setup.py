#!/usr/bin/env python3
"""
Setup script cho PDF Extract Multi-Agent System
Kiểm tra và cài đặt dependencies tự động
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command):
    """Chạy command và trả về kết quả"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_python_version():
    """Kiểm tra Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_requirements():
    """Cài đặt requirements"""
    print("📦 Cài đặt dependencies...")
    
    # Upgrade pip first
    success, stdout, stderr = run_command(f"{sys.executable} -m pip install --upgrade pip")
    if not success:
        print(f"⚠️ Warning: Could not upgrade pip: {stderr}")
    
    # Install requirements
    success, stdout, stderr = run_command(f"{sys.executable} -m pip install -r requirements.txt")
    if not success:
        print(f"❌ Error installing requirements: {stderr}")
        return False
    
    print("✅ Dependencies installed successfully")
    return True

def check_dependencies():
    """Kiểm tra conflicts trong dependencies"""
    print("🔍 Kiểm tra conflicts...")
    success, stdout, stderr = run_command(f"{sys.executable} -m pip check")
    if not success:
        print(f"⚠️ Dependency conflicts found: {stderr}")
        return False
    
    print("✅ No dependency conflicts found")
    return True

def create_directories():
    """Tạo các thư mục cần thiết"""
    print("📁 Tạo thư mục...")
    directories = ["uploads", "outputs", "temp_files", "indices"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✅ {directory}/")
    
    return True

def check_environment():
    """Kiểm tra environment variables"""
    print("🔍 Kiểm tra environment...")
    
    env_file = Path(".env")
    if not env_file.exists():
        print("ℹ️ Tạo file .env template...")
        with open(env_file, "w") as f:
            f.write("# OpenAI API Configuration\n")
            f.write("OPENAI_API_KEY=your_api_key_here\n")
            f.write("MODEL_NAME=gpt-4o-mini\n")
        print("  ✅ .env template created")
    else:
        print("  ✅ .env file exists")
    
    return True

def main():
    """Main setup function"""
    print("🚀 PDF Extract Multi-Agent System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        print("⚠️ Warning: Dependency conflicts detected")
    
    # Create directories
    create_directories()
    
    # Check environment
    check_environment()
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Add your OpenAI API key to .env file")
    print("2. Run: streamlit run streamlit_app.py")
    print("3. Or run: python main.py --help for CLI usage")

if __name__ == "__main__":
    main() 