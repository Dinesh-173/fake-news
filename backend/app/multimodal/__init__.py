"""TruthLens Multimodal Processing and Cross-Modal Consistency Package."""

from backend.app.multimodal.image_processor import (
    ImageProcessor,
    ImageValidationResult,
)
from backend.app.multimodal.ocr_engine import OCREngine, OCRResult
from backend.app.multimodal.vision_encoder import (
    MultimodalConsistencyResult,
    VisionEncoder,
)

__all__ = [
    "ImageProcessor",
    "ImageValidationResult",
    "MultimodalConsistencyResult",
    "OCREngine",
    "OCRResult",
    "VisionEncoder",
]
