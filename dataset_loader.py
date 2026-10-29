"""
Dataset Loader and Manager for CoMoFoD dataset and custom test samples.
"""

from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import cv2
import numpy as np


class CoMoFoDDataset:
    """
    Dataset loader for CoMoFoD image forgery dataset.
    Structure:
        dataset/
        ├── original/   (e.g., 001_O.png)
        ├── forged/     (e.g., 001_F.png)
        └── masks/      (e.g., 001_B.png)
    """

    def __init__(self, root_dir: str = "dataset"):
        self.root = Path(root_dir)
        self.orig_dir = self.root / "original"
        self.forged_dir = self.root / "forged"
        self.masks_dir = self.root / "masks"

        self.samples = self._discover_samples()

    def _discover_samples(self) -> List[Dict[str, Any]]:
        samples = []
        if not self.forged_dir.exists():
            return samples

        # Look for forged images
        forged_files = sorted(list(self.forged_dir.glob("*.png")) + list(self.forged_dir.glob("*.jpg")))

        for f_path in forged_files:
            stem = f_path.stem
            # In CoMoFoD, forged filenames typically end with _F or _F_*
            # Base ID extraction (e.g., '001_F' -> '001')
            parts = stem.split("_")
            sample_id = parts[0]

            # Find matching original
            orig_match = None
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = self.orig_dir / f"{sample_id}_O{ext}"
                if candidate.exists():
                    orig_match = candidate
                    break
                candidate_simple = self.orig_dir / f"{sample_id}{ext}"
                if candidate_simple.exists():
                    orig_match = candidate_simple
                    break

            # Find matching mask (_B for binary, or _M)
            mask_match = None
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate_b = self.masks_dir / f"{sample_id}_B{ext}"
                if candidate_b.exists():
                    mask_match = candidate_b
                    break
                candidate_m = self.masks_dir / f"{sample_id}_M{ext}"
                if candidate_m.exists():
                    mask_match = candidate_m
                    break
                candidate_stem = self.masks_dir / f"{stem}{ext}"
                if candidate_stem.exists():
                    mask_match = candidate_stem
                    break

            samples.append({
                "id": sample_id,
                "name": stem,
                "forged_path": str(f_path),
                "original_path": str(orig_match) if orig_match else None,
                "mask_path": str(mask_match) if mask_match else None,
            })

        return samples

    def get_sample_names(self) -> List[str]:
        return [s["name"] for s in self.samples]

    def get_sample(self, name_or_id: str) -> Optional[Dict[str, Any]]:
        for s in self.samples:
            if s["name"] == name_or_id or s["id"] == name_or_id:
                return s
        return None

    def load_sample_images(
        self,
        sample: Dict[str, Any],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load images as RGB numpy arrays.
        Returns:
            forged_rgb, original_rgb (or None), mask_gray (or None)
        """
        forged_rgb = self._read_rgb(sample["forged_path"])
        orig_rgb = self._read_rgb(sample["original_path"]) if sample.get("original_path") else None
        mask_gray = self._read_gray(sample["mask_path"]) if sample.get("mask_path") else None

        return forged_rgb, orig_rgb, mask_gray

    def _read_rgb(self, path: Optional[str]) -> Optional[np.ndarray]:
        if not path or not Path(path).exists():
            return None
        data = np.fromfile(path, dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _read_gray(self, path: Optional[str]) -> Optional[np.ndarray]:
        if not path or not Path(path).exists():
            return None
        data = np.fromfile(path, dtype=np.uint8)
        gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return None
        # Threshold to clean binary 0 or 255
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return binary


def generate_synthetic_sample(output_dir: str = "dataset/synthetic") -> Dict[str, str]:
    """
    Generate a clean synthetic copy-move sample with ground truth for testing/validation.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Base realistic textured canvas (512x512)
    h, w = 512, 512
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Gradient background
    y_grad = np.linspace(140, 220, h, dtype=np.uint8)[:, None]
    x_grad = np.linspace(120, 200, w, dtype=np.uint8)[None, :]
    img[:, :, 0] = (y_grad * 0.6 + x_grad * 0.4).astype(np.uint8)
    img[:, :, 1] = (y_grad * 0.3 + x_grad * 0.7).astype(np.uint8)
    img[:, :, 2] = 180

    # Draw complex natural-like objects (house, tree, star with rich ORB features)
    # Source object at (100, 120)
    def draw_complex_object(canvas, cx, cy):
        # Base rectangle
        cv2.rectangle(canvas, (cx - 40, cy - 30), (cx + 40, cy + 30), (50, 120, 200), -1)
        # Roof triangle
        pts = np.array([[cx - 50, cy - 30], [cx + 50, cy - 30], [cx, cy - 70]], np.int32)
        cv2.fillPoly(canvas, [pts], (180, 50, 50))
        # Window with crossbars (rich features)
        cv2.rectangle(canvas, (cx - 25, cy - 15), (cx - 5, cy + 5), (240, 240, 100), -1)
        cv2.line(canvas, (cx - 15, cy - 15), (cx - 15, cy + 5), (40, 40, 40), 1)
        cv2.line(canvas, (cx - 25, cy - 5), (cx - 5, cy - 5), (40, 40, 40), 1)
        # Door
        cv2.rectangle(canvas, (cx + 5, cy - 5), (cx + 25, cy + 30), (70, 40, 20), -1)
        # Circular ornament
        cv2.circle(canvas, (cx, cy - 45), 8, (230, 230, 80), -1)

    orig = img.copy()
    draw_complex_object(orig, 130, 260)
    # Extra static background object
    cv2.circle(orig, (400, 120), 45, (60, 160, 80), -1)
    cv2.circle(orig, (380, 100), 20, (80, 200, 100), -1)

    # Forged copy: clone object from (130, 260) to (320, 360)
    forged = orig.copy()
    # Mask of the cloned object
    mask = np.zeros((h, w), dtype=np.uint8)

    # Crop source patch (x: 70..190, y: 180..300)
    sx1, sy1, sx2, sy2 = 70, 180, 190, 300
    src_patch = orig[sy1:sy2, sx1:sx2].copy()

    # Paste destination (x: 270..390, y: 280..400)
    dx1, dy1 = 270, 280
    dx2, dy2 = dx1 + (sx2 - sx1), dy1 + (sy2 - sy1)

    # Insert copy into forged image
    forged[dy1:dy2, dx1:dx2] = src_patch

    # Ground truth mask marks both source and destination copy-move regions
    mask[sy1:sy2, sx1:sx2] = 255
    mask[dy1:dy2, dx1:dx2] = 255

    orig_p = str(out_path / "synthetic_01_O.png")
    forged_p = str(out_path / "synthetic_01_F.png")
    mask_p = str(out_path / "synthetic_01_B.png")

    cv2.imwrite(orig_p, cv2.cvtColor(orig, cv2.COLOR_RGB2BGR))
    cv2.imwrite(forged_p, cv2.cvtColor(forged, cv2.COLOR_RGB2BGR))
    cv2.imwrite(mask_p, mask)

    return {
        "id": "synthetic_01",
        "name": "synthetic_01_F",
        "original_path": orig_p,
        "forged_path": forged_p,
        "mask_path": mask_p,
    }
