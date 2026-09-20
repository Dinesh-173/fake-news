"""Multimodal availability, image asset verification, and modality breakdown module."""

import hashlib
from pathlib import Path

from pydantic import BaseModel, Field

from backend.app.dataset.schemas import CanonicalNewsRecord


class ImageDimensionStats(BaseModel):
    min_val: float = 0.0
    max_val: float = 0.0
    mean_val: float = 0.0
    median_val: float = 0.0


class ImageQualityProfile(BaseModel):
    """Detailed visual quality assessment when binary image files are locally present."""

    total_images_checked: int = 0
    valid_images: int = 0
    corrupted_images: int = 0
    unique_image_hashes: int = 0
    duplicate_image_count: int = 0
    format_distribution: dict[str, int] = Field(default_factory=dict)
    file_size_bytes_stats: ImageDimensionStats = Field(default_factory=ImageDimensionStats)
    width_stats: ImageDimensionStats = Field(default_factory=ImageDimensionStats)
    height_stats: ImageDimensionStats = Field(default_factory=ImageDimensionStats)


class ModalityAnalysisResult(BaseModel):
    """Rigorous breakdown of textual, visual, and metadata modalities across records."""

    total_records: int
    text_only_count: int
    text_only_percentage: float
    image_only_count: int
    image_only_percentage: float
    text_and_image_count: int
    text_and_image_percentage: float
    text_and_metadata_count: int
    text_and_metadata_percentage: float
    text_image_and_metadata_count: int
    text_image_and_metadata_percentage: float
    # Three-tier image availability reporting
    image_metadata_available: bool = Field(
        ..., description="Whether records reference remote or local image URLs/paths"
    )
    image_binary_available: bool = Field(
        ..., description="Whether local binary image files actually exist on the filesystem"
    )
    image_analysis_possible: bool = Field(
        ..., description="Whether local image files are decodable and structurally analyzable"
    )
    image_reference_count: int = Field(
        ..., description="Total records containing an image URL or image path"
    )
    image_local_file_count: int = Field(
        ..., description="Total records where image file was found locally on disk"
    )
    image_quality: ImageQualityProfile = Field(default_factory=ImageQualityProfile)
    notes: list[str] = Field(default_factory=list)


class ModalityAnalyzer:
    """Evaluates cross-modal availability and verifies the integrity of referenced image assets."""

    @classmethod
    def analyze(cls, records: list[CanonicalNewsRecord]) -> ModalityAnalysisResult:
        if not records:
            return ModalityAnalysisResult(
                total_records=0,
                text_only_count=0,
                text_only_percentage=0.0,
                image_only_count=0,
                image_only_percentage=0.0,
                text_and_image_count=0,
                text_and_image_percentage=0.0,
                text_and_metadata_count=0,
                text_and_metadata_percentage=0.0,
                text_image_and_metadata_count=0,
                text_image_and_metadata_percentage=0.0,
                image_metadata_available=False,
                image_binary_available=False,
                image_analysis_possible=False,
                image_reference_count=0,
                image_local_file_count=0,
                notes=["Empty dataset provided; no modality metrics computed."],
            )

        total = len(records)
        c_text_only = 0
        c_img_only = 0
        c_text_img = 0
        c_text_meta = 0
        c_text_img_meta = 0

        image_ref_count = 0
        local_image_paths: list[Path] = []

        for r in records:
            has_txt = bool(r.text and r.text.strip())
            has_img = bool(r.image_path or r.image_url)
            has_meta = bool(r.metadata and any(v for v in r.metadata.values() if v))

            if has_img:
                image_ref_count += 1
                if r.image_path:
                    p = Path(r.image_path)
                    if p.exists() and p.is_file():
                        local_image_paths.append(p)

            # Categorize modality combination
            if has_txt and not has_img and not has_meta:
                c_text_only += 1
            elif not has_txt and has_img and not has_meta:
                c_img_only += 1
            elif has_txt and has_img and not has_meta:
                c_text_img += 1
            elif has_txt and not has_img and has_meta:
                c_text_meta += 1
            elif has_txt and has_img and has_meta:
                c_text_img_meta += 1

        img_meta_avail = image_ref_count > 0
        img_bin_avail = len(local_image_paths) > 0
        img_analysis_possible = False

        quality_profile = ImageQualityProfile()

        # If binary images are genuinely present, profile them with Pillow safely
        if img_bin_avail:
            try:
                from PIL import Image

                widths: list[int] = []
                heights: list[int] = []
                sizes: list[int] = []
                formats: dict[str, int] = {}
                hashes: set[str] = set()
                corrupted = 0

                for p in local_image_paths:
                    try:
                        sizes.append(p.stat().st_size)
                        file_hash = hashlib.sha256(p.read_bytes()).hexdigest()
                        hashes.add(file_hash)
                        with Image.open(p) as img:
                            img.verify()
                        with Image.open(p) as img:
                            w, h = img.size
                            widths.append(w)
                            heights.append(h)
                            fmt = img.format or "UNKNOWN"
                            formats[fmt] = formats.get(fmt, 0) + 1
                    except Exception:
                        corrupted += 1

                img_analysis_possible = len(widths) > 0

                def calc_dim_stats(nums: list[int]) -> ImageDimensionStats:
                    if not nums:
                        return ImageDimensionStats()
                    import numpy as np

                    return ImageDimensionStats(
                        min_val=float(min(nums)),
                        max_val=float(max(nums)),
                        mean_val=round(float(np.mean(nums)), 2),
                        median_val=round(float(np.median(nums)), 2),
                    )

                quality_profile = ImageQualityProfile(
                    total_images_checked=len(local_image_paths),
                    valid_images=len(widths),
                    corrupted_images=corrupted,
                    unique_image_hashes=len(hashes),
                    duplicate_image_count=len(local_image_paths) - len(hashes),
                    format_distribution=formats,
                    file_size_bytes_stats=calc_dim_stats(sizes),
                    width_stats=calc_dim_stats(widths),
                    height_stats=calc_dim_stats(heights),
                )
            except Exception:
                img_analysis_possible = False

        notes: list[str] = []
        if not img_meta_avail:
            notes.append("Dataset is unimodal textual; no image references or metadata present.")
        elif img_meta_avail and not img_bin_avail:
            notes.append(
                f"Dataset contains {image_ref_count} image references/URLs, but 0 binary image files exist locally on disk."
            )
            notes.append(
                "Image analysis cannot be conducted until raw visual assets are acquired and hydrated."
            )
        else:
            notes.append(
                f"Verified {len(local_image_paths)} local image assets ({quality_profile.valid_images} valid, {quality_profile.corrupted_images} corrupted)."
            )

        return ModalityAnalysisResult(
            total_records=total,
            text_only_count=c_text_only,
            text_only_percentage=round((c_text_only / total) * 100, 2),
            image_only_count=c_img_only,
            image_only_percentage=round((c_img_only / total) * 100, 2),
            text_and_image_count=c_text_img,
            text_and_image_percentage=round((c_text_img / total) * 100, 2),
            text_and_metadata_count=c_text_meta,
            text_and_metadata_percentage=round((c_text_meta / total) * 100, 2),
            text_image_and_metadata_count=c_text_img_meta,
            text_image_and_metadata_percentage=round((c_text_img_meta / total) * 100, 2),
            image_metadata_available=img_meta_avail,
            image_binary_available=img_bin_avail,
            image_analysis_possible=img_analysis_possible,
            image_reference_count=image_ref_count,
            image_local_file_count=len(local_image_paths),
            image_quality=quality_profile,
            notes=notes,
        )
