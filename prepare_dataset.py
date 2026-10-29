"""
Script to extract and organize CoMoFoD dataset samples from comofod_small.rar into:
    dataset/original/
    dataset/forged/
    dataset/masks/
"""

import os
import shutil
import subprocess
from pathlib import Path

SEVEN_ZIP_PATH = r"C:\Program Files\7-Zip\7z.exe"
RAR_FILE = Path("comofod_small.rar")
DATASET_DIR = Path("dataset")
ORIGINAL_DIR = DATASET_DIR / "original"
FORGED_DIR = DATASET_DIR / "forged"
MASKS_DIR = DATASET_DIR / "masks"


def setup_directories():
    ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
    FORGED_DIR.mkdir(parents=True, exist_ok=True)
    MASKS_DIR.mkdir(parents=True, exist_ok=True)


def extract_comofod_samples(num_samples: int = 30):
    """
    Extract base sample sets: 001_O.png, 001_F.png, 001_B.png up to num_samples.
    """
    setup_directories()

    if not RAR_FILE.exists():
        print(f"Warning: {RAR_FILE} not found. Creating synthetic sample instead.")
        from dataset_loader import generate_synthetic_sample
        generate_synthetic_sample()
        return

    # Generate file patterns to extract
    files_to_extract = []
    for i in range(1, num_samples + 1):
        idx_str = f"{i:03d}"
        files_to_extract.extend([
            f"CoMoFoD_small_v2/{idx_str}_O.png",
            f"CoMoFoD_small_v2/{idx_str}_F.png",
            f"CoMoFoD_small_v2/{idx_str}_B.png",
            # Also extract some JPEG-compressed variants for ELA testing if present
            f"CoMoFoD_small_v2/{idx_str}_F_JC3.jpg",
            f"CoMoFoD_small_v2/{idx_str}_O_JC3.jpg",
        ])

    temp_extract = Path("dataset/_temp_extract")
    temp_extract.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {len(files_to_extract)} files from {RAR_FILE} using 7-Zip...")

    # Write files list to a response file for 7z
    list_file = temp_extract / "file_list.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for item in files_to_extract:
            f.write(f"{item}\n")

    cmd = [
        SEVEN_ZIP_PATH,
        "e",
        str(RAR_FILE),
        f"-o{temp_extract}",
        f"@{list_file}",
        "-y",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"7z command returned {result.returncode}: {result.stderr}")
    else:
        print("Extraction complete. Organizing into dataset folders...")

    # Organize extracted files into respective folders
    extracted_files = list(temp_extract.glob("*.*"))
    moved_count = 0

    for f in extracted_files:
        if f.name == "file_list.txt":
            continue
        stem = f.stem

        if "_O" in stem:
            target = ORIGINAL_DIR / f.name
            shutil.move(str(f), str(target))
            moved_count += 1
        elif "_F" in stem:
            target = FORGED_DIR / f.name
            shutil.move(str(f), str(target))
            moved_count += 1
        elif "_B" in stem or "_M" in stem:
            target = MASKS_DIR / f.name
            shutil.move(str(f), str(target))
            moved_count += 1

    # Cleanup temp folder
    shutil.rmtree(temp_extract, ignore_errors=True)

    print(f"Successfully organized {moved_count} images into {DATASET_DIR.resolve()}")


if __name__ == "__main__":
    extract_comofod_samples(num_samples=25)

