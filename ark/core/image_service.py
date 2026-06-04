#! python3
# -*- encoding: utf-8 -*-
"""
@File   :   image_service.py
@Created:   2026/06/05 00:18
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

# Here put the import lib.
import io
import base64
from typing import List, Optional

from PIL import Image as PILImage


def convert_image_to_data_url(image: PILImage.Image) -> str:
    """Convert a PIL image instance to a PNG data URL string.

    This method standardizes the output to PNG format, making it generic and
    unaffected by the source image's original format. PNG is chosen for its
    lossless quality and universal support for transparency (alpha channel),
    ensuring maximum compatibility with multimodal LLM inputs.

    Args:
        image (PIL.Image.Image): The input image to be encoded.

    Returns:
        image_data_url (str): Base64 data URL string (e.g. ``"data:image/png;base64,..."``).
    """
    # Standardize image mode to handle various source modes (like CMYK, LAB, etc.)
    # PNG natively supports RGB, RGBA, L (grayscale), and P (indexed).
    # If the mode is not among these, we convert to RGBA to ensure compatibility and preserve transparency.
    if image.mode not in ("RGB", "RGBA", "L", "P"):
        image = image.convert("RGBA")

    buf = io.BytesIO()
    # Save as PNG which is a robust, lossless format supported by all major AI APIs.
    image.save(buf, format="PNG")
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"
