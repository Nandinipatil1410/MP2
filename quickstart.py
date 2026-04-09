#!/usr/bin/env python3
"""
Quick Start Script
Privacy-Preserving Hybrid LLM System
"""
import subprocess
import sys
import os
from pathlib import Path


def print_header(text):
    print("\n" + "="*70)
    print(text)
    print("="*70 + "\n")


def check_python():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor} (Need 3.8+)")
        return False


def check_ollama():
    """Check if Ollama is installed"""
    print("Checking Ollama...")
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Ollama installed: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        print("✗ Ollama not found")
        print("  Install from: https://ollama.com/download")
        return False
    return False


def check_ollama_running():
    """Check if Ollama is running"""
    print("Checking if Ollama is running...")
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("✓ Ollama is running")
            models = response.json().get('models', [])
            if models:
                print(f"  Available models: {[m['name'] for m in models[:3]]}")
            return True
    except:
        print("✗ Ollama not running")
        print("  Start with: ollama serve")
        return False
    return False


def get_python_executable():
    """Get the path to the Python executable to use"""
    # Check if we're already in a venv
    if sys.prefix != sys.base_prefix:
        return sys.executable
    
    # Check if a venv directory exists locally
    venv_python = Path("./venv/bin/python")
    if venv_python.exists():
        return str(venv_python)
    
    return sys.executable


def install_dependencies():
    """Install Python dependencies"""
    print_header("INSTALLING DEPENDENCIES")
    print("This may take a few minutes...")
    
    python_exe = get_python_executable()
    
    try:
        subprocess.run([python_exe, '-m', 'pip', 'install', '-r', 'requirements.txt'],
                      check=True)
        print("✓ All dependencies installed")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install dependencies")
        if python_exe == sys.executable:
            print("\n💡 Tip: You may be in an externally managed environment.")
            print("   Try creating a virtual environment first:")
            print("   python3 -m venv venv")
            print("   Then run this script again.")
        return False


def setup_directories():
    """Create necessary directories"""
    print("Setting up directories...")
    dirs = [
        Path("./data/documents"),
        Path("./data/vector_db"),
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {dir_path}")


def create_env_file():
    """Create .env file if it doesn't exist"""
    env_path = Path(".env")
    if not env_path.exists():
        print("Creating .env file...")
        example_path = Path(".env.example")
        if example_path.exists():
            with open(example_path, 'r') as src:
                content = src.read()
            with open(env_path, 'w') as dst:
                dst.write(content)
            print("✓ Created .env file")
            print("  Edit .env to add your Groq API key (optional)")
        else:
            print("⚠️  .env.example not found")
    else:
        print("✓ .env file already exists")


def pull_ollama_model():
    """Pull a model for Ollama"""
    print("\nWould you like to pull the 'mistral' model for Ollama?")
    print("This will download ~4GB and is required for local execution.")
    response = input("Pull model? (y/n): ").lower()
    
    if response == 'y':
        print("Pulling mistral model...")
        print("This may take several minutes depending on your internet speed...")
        try:
            subprocess.run(['ollama', 'pull', 'mistral'], check=True)
            print("✓ Model downloaded successfully")
            return True
        except subprocess.CalledProcessError:
            print("✗ Failed to pull model")
            return False
    else:
        print("Skipped model download")
        return False


def main():
    print_header("PRIVACY-PRESERVING HYBRID LLM SYSTEM - QUICK START")
    
    # Check prerequisites
    print_header("CHECKING PREREQUISITES")
    
    python_ok = check_python()
    ollama_ok = check_ollama()
    
    if not python_ok:
        print("\n❌ Python 3.8+ is required. Please upgrade Python.")
        return
    
    if not ollama_ok:
        print("\n⚠️  Ollama not installed")
        print("Install from: https://ollama.com/download")
        print("Then run this script again")
        return
    
    ollama_running = check_ollama_running()
    
    # Setup
    print_header("SETUP")
    
    setup_directories()
    create_env_file()
    
    # Install dependencies
    print("\nInstall Python dependencies?")
    response = input("Continue? (y/n): ").lower()
    
    if response == 'y':
        if not install_dependencies():
            print("\n❌ Setup failed")
            return
    else:
        print("Skipped dependency installation")
    
    # Pull model
    if ollama_running:
        pull_ollama_model()
    else:
        print("\n⚠️  Start Ollama first: ollama serve")
        print("Then pull a model: ollama pull mistral")
    
    # Final instructions
    print_header("SETUP COMPLETE! 🎉")
    
    python_exe = "python3"
    if Path("./venv/bin/python").exists():
        python_exe = "./venv/bin/python"
    
    print("Next steps:")
    print("\n1. Start Ollama (if not running):")
    print("   ollama serve")
    
    print("\n2. Run the demo:")
    print(f"   {python_exe} test_demo.py --full")
    
    print("\n3. Or start the web interface (requires source venv/bin/activate first):")
    print("   streamlit run src/app.py")
    
    print("\n4. Access the UI:")
    print("   http://localhost:8501")
    
    print("\n📚 Documentation:")
    print("   - Setup Guide: SETUP.md")
    print("   - Full Docs: PROJECT_DOCUMENTATION.md")
    
    print("\n💡 Tips:")
    print("   - Groq API key is optional (system has fallback)")
    print("   - Add API key in .env or web interface")
    print("   - All documents stay on your machine")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("Please check the SETUP.md file for manual installation steps")
