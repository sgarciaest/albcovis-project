import requests
from pathlib import Path
from PIL import Image
from io import BytesIO
from urllib.parse import urlparse
from typing import Optional
import numpy as np
from skimage import color, util


def download_image(url: str, save_dir: str, filename: Optional[str] = None) -> Path:
    """
    Download an image from a URL and save it to the specified directory.
    If filename is not provided, the original filename from the URL is used.

    Overwrites existing files.
    """
    save_dir = Path(save_dir)
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Could not create directory {save_dir}: {e}")
        raise

    # If no filename given, extract from URL path
    if filename is None:
        parsed_url = urlparse(url)
        filename = Path(parsed_url.path).name or "downloaded_image"

    save_path = save_dir / filename

    if save_path.exists():
        print(f"[WARNING] File {save_path} already exists and will be overwritten.")

    try:
        with requests.get(url, stream=True, timeout=10) as response:
            response.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Avoid writing keep-alive chunks
                        f.write(chunk)
        print(f"Image saved to {save_path}")
    except requests.RequestException as e:
        print(f"Failed to download image from {url}: {e}")
        raise RuntimeError(f"Failed to download image from {url}") from e
    except OSError as e:
        print(f"Could not write image to {save_path}: {e}")
        raise

    return save_path

def pil_to_numpy01(img: Image.Image) -> np.ndarray:
    """
    Convert a PIL RGB image to a float32 NumPy array in [0, 1].
    Output shape: (H, W, 3).
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    # skimage works best with float in [0,1]
    rgb01 = util.img_as_float32(arr)  # scales uint8 [0,255] -> float32 [0,1]
    return rgb01


def rgb_to_gray(rgb01: np.ndarray) -> np.ndarray:
    """
    Convert an RGB float image in [0,1] to grayscale float in [0,1].
    """
    # skimage.color.rgb2gray expects float in [0,1]; returns float in [0,1]
    gray = color.rgb2gray(rgb01)
    gray01 = gray.astype(np.float32)
    return gray01


def limit_image_size(img: Image.Image, target_area: int = 512*512) -> Image.Image:
    """
    Downsample an image so that its area does not exceed target_area pixels.
    Keeps aspect ratio. If the image is already small enough, returns it unchanged.

    Args:
        img: A PIL.Image.Image object in RGB mode.
        target_area: Maximum allowed area in pixels (default: 512*512).

    Returns:
        A resized PIL.Image.Image object (or the original if no resize needed).
    """
    w, h = img.size
    area = w * h
    scale = target_area / area if area > target_area else 1.0

    if scale < 1.0:
        new_w = max(1, int(w * scale**0.5))
        new_h = max(1, int(h * scale**0.5))
        return img.resize((new_w, new_h), Image.BILINEAR)
    else:
        return img
