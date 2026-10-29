"""
Image preprocessing routines: loading, resizing, grayscale conversion, and noise filtering.
"""

from typing import Tuple, Union, Optional
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import io


def load_image(source: Union[str, Path, bytes, io.BytesIO, np.ndarray, Image.Image]) -> np.ndarray:
    """
    Load an image from various sources (file path, raw bytes, BytesIO, PIL Image, or numpy array)
    and return an RGB uint8 numpy array.
    """
    if isinstance(source, np.ndarray):
        if source.ndim == 2:
            return cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
        elif source.shape[2] == 4:
            return cv2.cvtColor(source, cv2.COLOR_RGBA2RGB)
        elif source.shape[2] == 3:
            return source.copy()
        else:
            raise ValueError(f"Unsupported numpy image shape: {source.shape}")

    if isinstance(source, Image.Image):
        rgb_img = source.convert("RGB")
        return np.array(rgb_img)

    if isinstance(source, (bytes, bytearray)):
        nparr = np.frombuffer(source, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Failed to decode image from byte buffer.")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if isinstance(source, io.BytesIO):
        source.seek(0)
        return load_image(source.read())

    # Path or string
    path_obj = Path(source)
    if not path_obj.exists():
        raise FileNotFoundError(f"Image file not found: {source}")

    # Use cv2.imdecode with np.fromfile to handle non-ASCII/Windows paths safely
    img_data = np.fromfile(str(path_obj), dtype=np.uint8)
    bgr = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image from {path_obj}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_image(
    image: np.ndarray,
    max_dim: Optional[int] = 1000,
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Resize image to either a target size (width, height) or constrain the maximum
    dimension (width or height) to max_dim while preserving aspect ratio.

    Returns:
        resized_image: resized RGB or grayscale image
        scale_x: width scaling factor (resized_w / orig_w)
        scale_y: height scaling factor (resized_h / orig_h)
    """
    orig_h, orig_w = image.shape[:2]

    if target_size is not None:
        target_w, target_h = target_size
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return resized, target_w / orig_w, target_h / orig_h

    if max_dim is not None and max(orig_h, orig_w) > max_dim:
        scale = max_dim / float(max(orig_h, orig_w))
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, new_w / orig_w, new_h / orig_h

    # No resize needed
    return image.copy(), 1.0, 1.0


def to_grayscale(image_rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to single-channel 8-bit grayscale.
    """
    if image_rgb.ndim == 2:
        return image_rgb.copy()
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)


def reduce_noise(
    image_gray: np.ndarray,
    method: str = "gaussian",
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Apply smoothing / noise reduction to a grayscale image.

    Args:
        image_gray: grayscale uint8 image
        method: 'gaussian', 'median', or 'bilateral'
        kernel_size: kernel size (must be odd integer >= 1)
        sigma: Gaussian sigma or filter intensity

    Returns:
        filtered grayscale uint8 image
    """
    if kernel_size <= 1:
        return image_gray.copy()

    # Ensure odd kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1

    method_lower = method.lower()
    if method_lower == "gaussian":
        return cv2.GaussianBlur(image_gray, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)
    elif method_lower == "median":
        return cv2.medianBlur(image_gray, kernel_size)
    elif method_lower == "bilateral":
        return cv2.bilateralFilter(image_gray, d=kernel_size, sigmaColor=75, sigmaSpace=75)
    else:
        return cv2.GaussianBlur(image_gray, (kernel_size, kernel_size), sigmaX=sigma)


def preprocess_pipeline(
    image_source: Union[str, Path, bytes, io.BytesIO, np.ndarray, Image.Image],
    max_dim: Optional[int] = 1000,
    noise_reduction: str = "gaussian",
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> dict:
    """
    Execute full preprocessing pipeline.

    Returns dict containing:
        original_rgb: untouched loaded image
        resized_rgb: image scaled according to max_dim
        gray: grayscale of resized image
        filtered_gray: noise-reduced grayscale image
        scale_x, scale_y: scaling factors relative to original
    """
    original_rgb = load_image(image_source)
    resized_rgb, sx, sy = resize_image(original_rgb, max_dim=max_dim)
    gray = to_grayscale(resized_rgb)
    filtered = reduce_noise(gray, method=noise_reduction, kernel_size=kernel_size, sigma=sigma)

    return {
        "original_rgb": original_rgb,
        "resized_rgb": resized_rgb,
        "gray": gray,
        "filtered_gray": filtered,
        "scale_x": sx,
        "scale_y": sy,
    }

