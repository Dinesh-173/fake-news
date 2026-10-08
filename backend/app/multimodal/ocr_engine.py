"""TruthLens Optical Character Recognition (OCR) Engine.

Extracts embedded textual assertions from memes, infographics, screenshots, and news graphics.
Includes robust fallback when system Tesseract binary is unavailable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

logger = logging.getLogger("truthlens.multimodal.ocr")


@dataclass
class OCRResult:
    ocr_text: str
    char_count: int
    word_count: int
    is_ocr_available: bool
    confidence: float
    extracted_entities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ocr_text": self.ocr_text,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "is_ocr_available": self.is_ocr_available,
            "confidence": round(self.confidence, 3),
            "extracted_entities": self.extracted_entities,
        }


class OCREngine:
    """Engine for extracting text from visual assets."""

    @classmethod
    def extract_text(cls, pil_image: Image.Image | None) -> OCRResult:
        """
        Extracts embedded textual content from a PIL image.
        Falls back gracefully if Tesseract binary is not installed on system.
        """
        if pil_image is None:
            return OCRResult(
                ocr_text="",
                char_count=0,
                word_count=0,
                is_ocr_available=False,
                confidence=0.0,
            )

        try:
            import pytesseract

            # Convert to RGB if needed
            if pil_image.mode not in ("RGB", "L"):
                pil_image = pil_image.convert("RGB")

            raw_text = pytesseract.image_to_string(pil_image)
            clean_text = " ".join(raw_text.split()).strip()

            words = clean_text.split()
            # Heuristic entity extraction from capitalized tokens
            entities = list(set(re.findall(r'\b[A-Z][a-z0-9]+\b', clean_text)))

            confidence = 0.85 if len(words) >= 3 else (0.50 if len(words) > 0 else 0.0)

            return OCRResult(
                ocr_text=clean_text,
                char_count=len(clean_text),
                word_count=len(words),
                is_ocr_available=True,
                confidence=confidence,
                extracted_entities=entities[:10],
            )

        except Exception as exc:
            # Handles pytesseract.TesseractNotFoundError or execution errors
            logger.debug("Tesseract OCR unavailable or failed: %s", exc)
            return OCRResult(
                ocr_text="",
                char_count=0,
                word_count=0,
                is_ocr_available=False,
                confidence=0.0,
            )
