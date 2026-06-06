#! python3
# -*- encoding: utf-8 -*-
"""
@File   :   image_service.py
@Created:   2026/03/06 12:00
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

# Here put the import lib.
import io
import base64
from typing import List, Optional, Tuple

from PIL import Image as PILImage


def __encode_image_to_png_base64(image: PILImage.Image) -> str:
    """Standardize a PIL image to PNG and return its base64-encoded payload.

    PNG is chosen for its lossless quality and universal support for
    transparency (alpha channel), ensuring maximum compatibility with
    multimodal LLM inputs regardless of the source image's original format.

    Args:
        image (PIL.Image.Image): The input image to be encoded.

    Returns:
        image_base64 (str): The base64-encoded PNG payload (no data-URL prefix).
    """
    # Standardize image mode to handle various source modes (like CMYK, LAB, etc.)
    # PNG natively supports RGB, RGBA, L (grayscale), and P (indexed).
    # If the mode is not among these, we convert to RGBA to ensure compatibility and preserve transparency.
    if image.mode not in ("RGB", "RGBA", "L", "P"):
        image = image.convert("RGBA")

    buf = io.BytesIO()
    # Save as PNG which is a robust, lossless format supported by all major AI APIs.
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def convert_image_to_data_url(image: PILImage.Image) -> str:
    """Convert a PIL image instance to a PNG data URL string.

    This method standardizes the output to PNG format, making it generic and
    unaffected by the source image's original format. Used by OpenAI-compatible
    multimodal inputs, which expect an ``image_url`` data URL.

    Args:
        image (PIL.Image.Image): The input image to be encoded.

    Returns:
        image_data_url (str): Base64 data URL string (e.g. ``"data:image/png;base64,..."``).
    """
    return f"data:image/png;base64,{__encode_image_to_png_base64(image)}"


def convert_image_to_base64(image: PILImage.Image) -> Tuple[str, str]:
    """Convert a PIL image instance to a PNG media type and base64 payload.

    Used by APIs (such as Anthropic Messages) that expect the media type and
    the raw base64 data as separate fields rather than a combined data URL.

    Args:
        image (PIL.Image.Image): The input image to be encoded.

    Returns:
        A tuple of ``(media_type, image_base64)`` where ``media_type`` is
        ``"image/png"`` and ``image_base64`` is the base64-encoded PNG payload.
    """
    return "image/png", __encode_image_to_png_base64(image)
