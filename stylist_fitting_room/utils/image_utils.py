"""Image processing utilities."""

import base64
import io
import logging
from typing import Optional, Tuple
from urllib.parse import urlparse

import requests
from PIL import Image

logger = logging.getLogger(__name__)


def download_image(
    url: str,
    timeout: int = 10,
    max_size: Tuple[int, int] = (1024, 1024),
) -> Optional[Image.Image]:
    """
    Download an image from a URL and return as PIL Image.

    Args:
        url: URL of the image to download
        timeout: Request timeout in seconds
        max_size: Maximum dimensions to resize to (width, height)

    Returns:
        PIL Image or None if download failed
    """
    try:
        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme in ("http", "https"):
            logger.warning(f"Invalid URL scheme: {url}")
            return None

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        response = requests.get(url, timeout=timeout, headers=headers, stream=True)
        response.raise_for_status()

        # Check content type
        content_type = response.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            logger.warning(f"Non-image content type: {content_type}")
            return None

        # Load image
        image = Image.open(io.BytesIO(response.content))

        # Convert to RGB if necessary
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")

        # Resize if too large
        if image.width > max_size[0] or image.height > max_size[1]:
            image = resize_image(image, max_size)

        return image

    except requests.RequestException as e:
        logger.error(f"Failed to download image from {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error processing image from {url}: {e}")
        return None


def resize_image(
    image: Image.Image,
    max_size: Tuple[int, int] = (1024, 1024),
    keep_aspect: bool = True,
) -> Image.Image:
    """
    Resize an image to fit within max_size while preserving aspect ratio.

    Args:
        image: PIL Image to resize
        max_size: Maximum dimensions (width, height)
        keep_aspect: Whether to preserve aspect ratio

    Returns:
        Resized PIL Image
    """
    if keep_aspect:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        return image
    else:
        return image.resize(max_size, Image.Resampling.LANCZOS)


def pil_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """
    Convert a PIL Image to base64 string.

    Args:
        image: PIL Image
        format: Image format (JPEG, PNG)

    Returns:
        Base64 encoded string
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def base64_to_pil(base64_string: str) -> Image.Image:
    """
    Convert a base64 string to PIL Image.

    Args:
        base64_string: Base64 encoded image string

    Returns:
        PIL Image
    """
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))


def validate_image(image: Image.Image) -> bool:
    """
    Validate that an image is suitable for processing.

    Args:
        image: PIL Image to validate

    Returns:
        True if valid, False otherwise
    """
    if image is None:
        return False

    # Check minimum size
    if image.width < 100 or image.height < 100:
        logger.warning(f"Image too small: {image.width}x{image.height}")
        return False

    # Check maximum size
    if image.width > 4096 or image.height > 4096:
        logger.warning(f"Image too large: {image.width}x{image.height}")
        return False

    return True


def create_placeholder_image(
    width: int = 512,
    height: int = 512,
    color: Tuple[int, int, int] = (200, 200, 200),
    text: str = "No Image",
) -> Image.Image:
    """
    Create a placeholder image with text.

    Args:
        width: Image width
        height: Image height
        color: Background color (RGB)
        text: Text to display

    Returns:
        PIL Image
    """
    from PIL import ImageDraw

    image = Image.new("RGB", (width, height), color)
    draw = ImageDraw.Draw(image)

    # Draw text in center
    text_bbox = draw.textbbox((0, 0), text)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    x = (width - text_width) // 2
    y = (height - text_height) // 2

    draw.text((x, y), text, fill=(100, 100, 100))

    return image
