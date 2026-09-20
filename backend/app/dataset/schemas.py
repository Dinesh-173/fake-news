"""Canonical schemas for dataset records, validation summaries, and manifests."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CanonicalNewsRecord(BaseModel):
    """Canonical representation of a single news article/claim record across all datasets."""

    model_config = ConfigDict(extra="ignore")

    # Identifiers & Provenance
    record_id: str = Field(..., description="Unique canonical record UUID")
    dataset_name: str = Field(..., description="Source dataset (e.g. 'liar', 'fakenewsnet')")
    original_id: str = Field(..., description="Original identifier from the source dataset")

    # Core Textual Modality
    text: str = Field(..., description="Raw or cleaned body/statement text")
    title: str | None = Field(None, description="Headline or title if available")

    # Visual Modality
    image_path: str | None = Field(None, description="Local path to image file if present")
    image_url: str | None = Field(None, description="Remote URL of the original image")

    # Source & Attribution
    source_name_raw: str | None = Field(None, description="Original source string")
    source_name_normalized: str | None = Field(None, description="Canonical publisher/speaker")
    source_domain: str | None = Field(None, description="Extracted domain (e.g. 'cnn.com')")
    source_url: str | None = Field(None, description="Canonical source URL")
    author: str | None = Field(None, description="Author or speaker name")

    # Temporal Modality
    published_at_raw: str | None = Field(None, description="Raw timestamp string from dataset")
    published_at_normalized: datetime | None = Field(
        None, description="Standardized UTC publication timestamp"
    )

    # Labels & Normalization
    original_label: str = Field(..., description="Original raw label from source dataset")
    normalized_label: str | None = Field(
        None, description="Mapped TruthLens label: LIKELY_TRUE, LIKELY_FALSE, etc."
    )
    label_mapping_status: str = Field(
        default="mapped", description="'mapped', 'unresolved', or 'excluded'"
    )
    language: str = Field(default="en", description="Language code")

    # Extensible Metadata (e.g., credit history in LIAR, tweet id in FakeNewsNet)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Validation & Quality
    validation_status: str = Field(default="VALID", description="'VALID', 'WARNING', or 'INVALID'")
    validation_notes: list[str] = Field(default_factory=list)

    # Modality Availability Flags (Explicitly computed)
    has_text: bool = True
    has_image: bool = False
    has_metadata: bool = False
    has_timestamp: bool = False
    has_source: bool = False

    # Deduplication & Leakage Prevention
    content_hash: str | None = Field(
        None, description="SHA-256 hash of normalized text for exact deduplication"
    )
    duplicate_group_id: str | None = Field(
        None, description="Identifier of the cluster of identical records"
    )
    is_duplicate: bool = False
    duplicate_of: str | None = Field(None, description="Record ID of the retained primary copy")
    near_duplicate_group_id: str | None = Field(
        None, description="Identifier of near-duplicate cluster"
    )

    # Experimental Partition
    split: str | None = Field(None, description="'train', 'val', or 'test'")


class ValidationSummary(BaseModel):
    total_records: int
    valid_records: int
    warning_records: int
    invalid_records: int
    warning_reasons: dict[str, int] = Field(default_factory=dict)
    invalid_reasons: dict[str, int] = Field(default_factory=dict)


class ModalityStatistics(BaseModel):
    total: int
    text_available: int
    image_available: int
    text_and_image_available: int
    metadata_available: int
    timestamp_available: int
    source_available: int


class SplitStatistics(BaseModel):
    train_count: int
    val_count: int
    test_count: int
    train_distribution: dict[str, int] = Field(default_factory=dict)
    val_distribution: dict[str, int] = Field(default_factory=dict)
    test_distribution: dict[str, int] = Field(default_factory=dict)


class DuplicateStatistics(BaseModel):
    total_records: int
    unique_content_hashes: int
    exact_duplicates_count: int
    exact_duplicate_groups: int
    near_duplicate_groups: int


class DatasetManifest(BaseModel):
    dataset_name: str
    dataset_version: str
    pipeline_version: str = "1.0.0"
    created_at: str
    source_url_or_origin: str
    license_or_terms: str
    record_counts: dict[str, int]
    label_distribution_original: dict[str, int]
    label_distribution_normalized: dict[str, int]
    modality_statistics: ModalityStatistics
    duplicate_statistics: DuplicateStatistics
    split_statistics: SplitStatistics
    random_seed: int
    git_commit: str
    file_hashes: dict[str, str] = Field(default_factory=dict)
