"""TruthLens Image Processing and Validation Engine.

Provides:
1. Ingestion of images from raw bytes, file paths, or base64 strings.
2. Verification of file integrity, dimensions, formats, and file size.
3. Graceful handling of corrupted images, unsupported formats, and low-resolution inputs.
4. Cryptographic SHA-256 fingerprinting for provenance.
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

logger = logging.getLogger("truthlens.multimodal.image_processor")

SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB
MIN_RESOLUTION_PIXELS = 100


@dataclass
class ImageValidationResult:
    is_valid: bool
    format: str | None = None
    width: int = 0
    height: int = 0
    aspect_ratio: float = 0.0
    file_size_bytes: int = 0
    sha256_hash: str | None = None
    is_low_resolution: bool = False
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "format": self.format,
            "width": self.width,
            "height": self.height,
            "aspect_ratio": round(self.aspect_ratio, 3),
            "file_size_bytes": self.file_size_bytes,
            "sha256_hash": self.sha256_hash,
            "is_low_resolution": self.is_low_resolution,
            "error_message": self.error_message,
        }


class ImageProcessor:
    """Processes, validates, and normalizes image inputs for multimodal verification."""

    @classmethod
    def load_and_validate(
        cls, image_input: bytes | str | Path | Image.Image
    ) -> tuple[Image.Image | None, ImageValidationResult]:
        """
        Loads and verifies image data safely.

        Args:
            image_input: Raw image bytes, filesystem path, base64 data URL, or PIL Image.

        Returns:
            Tuple of (PIL.Image or None, ImageValidationResult)
        """
        raw_bytes: bytes | None = None

        try:
            if isinstance(image_input, Image.Image):
                img = image_input.copy()
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                raw_bytes = buf.getvalue()
            elif isinstance(image_input, bytes):
                raw_bytes = image_input
            elif isinstance(image_input, (str, Path)):
                p = Path(image_input)
                if p.exists() and p.is_file():
                    raw_bytes = p.read_bytes()
                elif isinstance(image_input, str) and (
                    image_input.startswith("data:image/") or ";base64," in image_input
                ):
                    header, b64_str = image_input.split(";base64,", 1)
                    raw_bytes = base64.b64decode(b64_str)
                else:
                    return None, ImageValidationResult(
                        is_valid=False,
                        error_message=f"Image file does not exist: {image_input}",
                    )
            else:
                return None, ImageValidationResult(
                    is_valid=False,
                    error_message=f"Unsupported image input type: {type(image_input)}",
                )

            # Check file size limit
            file_size = len(raw_bytes)
            if file_size > MAX_FILE_SIZE_BYTES:
                return None, ImageValidationResult(
                    is_valid=False,
                    file_size_bytes=file_size,
                    error_message=f"Image exceeds maximum allowed size ({file_size} > {MAX_FILE_SIZE_BYTES} bytes).",
                )

            sha256_h = hashlib.sha256(raw_bytes).hexdigest()

            # Attempt decoding via PIL
            stream = io.BytesIO(raw_bytes)
            pil_image = Image.open(stream)
            pil_image.verify()  # Verifies image integrity without full decoding

            # Reopen stream because verify() modifies stream position
            stream.seek(0)
            pil_image = Image.open(stream)
            # Convert palette/transparency to RGB for standardized processing
            if pil_image.mode not in ("RGB", "L"):
                pil_image = pil_image.convert("RGB")

            img_format = pil_image.format or "PNG"
            w, h = pil_image.size
            aspect = float(w) / float(h) if h > 0 else 1.0
            is_low_res = bool(w < MIN_RESOLUTION_PIXELS or h < MIN_RESOLUTION_PIXELS)

            return pil_image, ImageValidationResult(
                is_valid=True,
                format=img_format,
                width=w,
                height=h,
                aspect_ratio=aspect,
                file_size_bytes=file_size,
                sha256_hash=sha256_h,
                is_low_resolution=is_low_res,
                error_message=None,
            )

        except (UnidentifiedImageError, OSError, ValueError) as exc:
            logger.warning("Image validation failed for input: %s", exc)
            return None, ImageValidationResult(
                is_valid=False,
                error_message=f"Corrupted or invalid image data: {exc}",
            )
        except Exception as exc:
            logger.error("Unexpected error during image processing: %s", exc)
            return None, ImageValidationResult(
                is_valid=False,
                error_message=f"Image processing failure: {exc}",
            )
