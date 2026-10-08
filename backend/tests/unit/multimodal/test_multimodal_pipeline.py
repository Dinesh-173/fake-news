"""Unit tests for Phase 7: Multimodal Analysis, Image Processing, OCR, and Consistency."""

import io

from PIL import Image

from backend.app.multimodal.image_processor import ImageProcessor
from backend.app.multimodal.ocr_engine import OCREngine
from backend.app.multimodal.vision_encoder import (
    VISUAL_EMBEDDING_DIM,
    VisionEncoder,
)


def _create_sample_image(width: int = 200, height: int = 200, color: tuple = (255, 0, 0)) -> bytes:
    """Helper to generate in-memory PNG bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_image_processor_valid_image():
    """Verify image loading, dimension extraction, format verification, and SHA256 fingerprinting."""
    png_bytes = _create_sample_image(width=300, height=150)
    pil_img, val_result = ImageProcessor.load_and_validate(png_bytes)

    assert pil_img is not None
    assert val_result.is_valid is True
    assert val_result.format in ("PNG", "JPEG")
    assert val_result.width == 300
    assert val_result.height == 150
    assert abs(val_result.aspect_ratio - 2.0) < 1e-2
    assert val_result.is_low_resolution is False
    assert val_result.sha256_hash is not None
    assert len(val_result.sha256_hash) == 64


def test_image_processor_corrupted_bytes_handling():
    """Verify that corrupt or random bytes fail gracefully without crashing."""
    corrupted_data = b"NOT_A_VALID_IMAGE_HEADER_DATA_STREAM"
    pil_img, val_result = ImageProcessor.load_and_validate(corrupted_data)

    assert pil_img is None
    assert val_result.is_valid is False
    assert "Corrupted or invalid" in val_result.error_message


def test_image_processor_low_resolution_flag():
    """Verify that images below 100x100 resolution are flagged."""
    tiny_bytes = _create_sample_image(width=50, height=50)
    pil_img, val_result = ImageProcessor.load_and_validate(tiny_bytes)

    assert pil_img is not None
    assert val_result.is_valid is True
    assert val_result.is_low_resolution is True


def test_ocr_engine_execution_and_fallback():
    """Verify OCR engine runs safely and returns structured output."""
    sample_bytes = _create_sample_image()
    pil_img, _ = ImageProcessor.load_and_validate(sample_bytes)

    ocr_res = OCREngine.extract_text(pil_img)
    assert isinstance(ocr_res.ocr_text, str)
    assert isinstance(ocr_res.is_ocr_available, bool)
    assert 0.0 <= ocr_res.confidence <= 1.0


def test_vision_encoder_dense_embeddings():
    """Verify vision encoder produces normalized 128-dimensional embedding vector."""
    sample_bytes = _create_sample_image(width=128, height=128)
    pil_img, _ = ImageProcessor.load_and_validate(sample_bytes)

    vec = VisionEncoder.encode_image(pil_img)
    assert vec.shape == (VISUAL_EMBEDDING_DIM,)
    # Should be unit normalized
    norm = float(vec.dot(vec))
    assert abs(norm - 1.0) < 1e-3

    # Fallback when no image provided
    zero_vec = VisionEncoder.encode_image(None)
    assert zero_vec.shape == (VISUAL_EMBEDDING_DIM,)
    assert (zero_vec == 0.0).all()


def test_cross_modal_consistency_no_image_mode():
    """Verify graceful handling for text-only claims."""
    res = VisionEncoder.evaluate_cross_modal_consistency(
        claim_text="NASA confirms space satellite mission",
        pil_image=None,
    )
    assert res.has_image is False
    assert res.consistency_state == "NO_IMAGE"
    assert res.alignment_score == 1.0


def test_cross_modal_consistency_with_matching_ocr():
    """Verify alignment state when OCR text matches claim text."""
    sample_bytes = _create_sample_image()
    pil_img, _ = ImageProcessor.load_and_validate(sample_bytes)

    res_matching = VisionEncoder.evaluate_cross_modal_consistency(
        claim_text="NASA budget increases for Mars exploration",
        pil_image=pil_img,
        ocr_text="NASA budget report confirms funding increase for Mars",
    )
    assert res_matching.has_image is True
    assert res_matching.consistency_state == "ALIGNED"
    assert res_matching.alignment_score >= 0.60
