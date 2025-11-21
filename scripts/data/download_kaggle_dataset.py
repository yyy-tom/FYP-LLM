#!/usr/bin/env python3
"""
Script to download the Mental Health Counseling Conversations dataset from Kaggle.
Requires Kaggle API credentials to be set up.
"""

import os
import argparse
from pathlib import Path
import subprocess
import sys


def check_kaggle_installed():
    """Check if Kaggle API is installed."""
    try:
        import kaggle
        return True
    except ImportError:
        return False


def install_kaggle():
    """Install Kaggle API."""
    print("Installing Kaggle API...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
    print("Kaggle API installed successfully!")


def check_kaggle_credentials():
    """Check if Kaggle credentials are set up."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_json.exists():
        print("\n" + "="*60)
        print("Kaggle credentials not found!")
        print("="*60)
        print("\nTo set up Kaggle API credentials:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Scroll down to 'API' section")
        print("3. Click 'Create New Token' to download kaggle.json")
        print("4. Place kaggle.json in ~/.kaggle/ directory")
        print("5. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
        print("\nAlternatively, you can set environment variables:")
        print("  export KAGGLE_USERNAME='your_username'")
        print("  export KAGGLE_KEY='your_api_key'")
        print("="*60 + "\n")
        return False
    
    # Check permissions
    import stat
    file_stat = os.stat(kaggle_json)
    if file_stat.st_mode & stat.S_IRWXG or file_stat.st_mode & stat.S_IRWXO:
        print(f"Warning: kaggle.json has incorrect permissions. Run: chmod 600 {kaggle_json}")
    
    return True


def download_dataset(dataset_name: str, output_dir: str, unzip: bool = True):
    """Download dataset from Kaggle."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        print(f"Downloading dataset: {dataset_name}")
        print(f"Output directory: {output_dir}")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Download dataset
        api.dataset_download_files(
            dataset_name,
            path=output_dir,
            unzip=unzip
        )
        
        print(f"\nDataset downloaded successfully to {output_dir}!")
        
        # List downloaded files
        files = list(Path(output_dir).glob("*"))
        print(f"\nDownloaded files:")
        for f in files:
            if f.is_file():
                print(f"  - {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)")
            elif f.is_dir():
                print(f"  - {f.name}/ (directory)")
        
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have accepted the dataset's terms of use on Kaggle")
        print("2. Check that your Kaggle credentials are correct")
        print("3. Verify the dataset name is correct")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download Mental Health Counseling Conversations dataset from Kaggle"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="melissamonfared/mental-health-counseling-conversations-k",
        help="Kaggle dataset name (format: username/dataset-name)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="kaggle_mental_health",
        help="Output directory for downloaded dataset"
    )
    parser.add_argument(
        "--no-unzip",
        action="store_true",
        help="Don't unzip the downloaded files"
    )
    parser.add_argument(
        "--install-kaggle",
        action="store_true",
        help="Install Kaggle API if not already installed"
    )
    
    args = parser.parse_args()
    
    # Check if Kaggle is installed
    if not check_kaggle_installed():
        if args.install_kaggle:
            install_kaggle()
        else:
            print("Kaggle API not found. Install it with:")
            print("  pip install kaggle")
            print("Or run with --install-kaggle flag")
            sys.exit(1)
    
    # Check credentials
    if not check_kaggle_credentials():
        # Check environment variables as fallback
        if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
            print("\nPlease set up Kaggle credentials before proceeding.")
            sys.exit(1)
    
    # Download dataset
    success = download_dataset(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        unzip=not args.no_unzip
    )
    
    if success:
        print(f"\nNext steps:")
        print(f"1. Inspect the downloaded files in {args.output_dir}")
        print(f"2. Run prepare_kaggle_dataset.py to process the data")
        print(f"   python scripts/data/prepare_kaggle_dataset.py --input_dir {args.output_dir}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

