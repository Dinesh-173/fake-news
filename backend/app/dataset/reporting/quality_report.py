"""Quality report generator producing structured, auditable Markdown documentation."""

from pathlib import Path

from backend.app.core.logging import logger
from backend.app.dataset.schemas import DatasetManifest


class QualityReporter:
    """Formats research-grade Markdown dataset quality reports from manifests."""

    @classmethod
    def generate_markdown_report(cls, manifest: DatasetManifest) -> str:
        """Generates comprehensive Markdown report from manifest telemetry."""
        total_records = manifest.record_counts.get("total", 0)

        # Build original labels table
        orig_labels_rows = []
        for lbl, count in sorted(manifest.label_distribution_original.items()):
            pct = (count / total_records * 100) if total_records > 0 else 0.0
            orig_labels_rows.append(f"| `{lbl}` | {count:,} | {pct:.2f}% |")
        orig_table = "\n".join(orig_labels_rows)

        # Build normalized labels table
        norm_labels_rows = []
        for lbl, count in sorted(manifest.label_distribution_normalized.items()):
            pct = (count / total_records * 100) if total_records > 0 else 0.0
            norm_labels_rows.append(f"| `{lbl}` | {count:,} | {pct:.2f}% |")
        norm_table = "\n".join(norm_labels_rows)

        # Modality stats
        ms = manifest.modality_statistics
        text_pct = (ms.text_available / total_records * 100) if total_records > 0 else 0.0
        img_pct = (ms.image_available / total_records * 100) if total_records > 0 else 0.0
        both_pct = (ms.text_and_image_available / total_records * 100) if total_records > 0 else 0.0
        meta_pct = (ms.metadata_available / total_records * 100) if total_records > 0 else 0.0
        ts_pct = (ms.timestamp_available / total_records * 100) if total_records > 0 else 0.0
        src_pct = (ms.source_available / total_records * 100) if total_records > 0 else 0.0

        # Split stats
        ss = manifest.split_statistics
        train_pct = (ss.train_count / total_records * 100) if total_records > 0 else 0.0
        val_pct = (ss.val_count / total_records * 100) if total_records > 0 else 0.0
        test_pct = (ss.test_count / total_records * 100) if total_records > 0 else 0.0

        # Duplicate stats
        ds = manifest.duplicate_statistics

        reassign_sec = ""
        if ss.reassigned_count > 0:
            reassign_rows = []
            for item in ss.reassignments:
                reassign_rows.append(
                    f"| `{item.get('original_id', '')}` | `{item.get('original_split', '')}` | "
                    f"`{item.get('final_split', '')}` | `{item.get('cluster_id', '')}` | "
                    f"{item.get('reassignment_reason', '')} |"
                )
            reassign_table = "\n".join(reassign_rows)
            reassign_sec = f"""
### 6.2 Cluster Isolation Reassignments (Leakage Defense)
* **Reassigned Records**: {ss.reassigned_count:,}
* **Affected Clusters**: {ss.affected_clusters_count:,}
* **Train ↔ Validation Conflicts**: {ss.train_val_conflicts:,}
* **Train ↔ Test Conflicts**: {ss.train_test_conflicts:,}
* **Validation ↔ Test Conflicts**: {ss.val_test_conflicts:,}

| Record ID | Original Split | Final Split | Cluster ID | Reason |
| :--- | :--- | :--- | :--- | :--- |
{reassign_table}
"""

        report = f"""# TruthLens Dataset Quality Report: {manifest.dataset_name.upper()} ({manifest.dataset_version})

## 1. Dataset Provenance & Metadata
* **Dataset Identifier**: `{manifest.dataset_name}`
* **Dataset Version**: `{manifest.dataset_version}`
* **Pipeline Version**: `{manifest.pipeline_version}`
* **Generation Timestamp**: `{manifest.created_at}`
* **Source / Origin**: `{manifest.source_url_or_origin}`
* **License / Terms of Use**: `{manifest.license_or_terms}`
* **Git Commit**: `{manifest.git_commit}`
* **Random Seed**: `{manifest.random_seed}`

---

## 2. Ingestion & Validation Summary
* **Total Records Ingested**: {total_records:,}
* **Valid Records**: {manifest.record_counts.get("valid", 0):,} ({(manifest.record_counts.get("valid", 0) / max(total_records, 1) * 100):.2f}%)
* **Warning Records**: {manifest.record_counts.get("warning", 0):,} ({(manifest.record_counts.get("warning", 0) / max(total_records, 1) * 100):.2f}%)
* **Invalid Records**: {manifest.record_counts.get("invalid", 0):,} ({(manifest.record_counts.get("invalid", 0) / max(total_records, 1) * 100):.2f}%)

---

## 3. Label Normalization & Distribution

### 3.1 Original Dataset Labels
| Original Label | Count | Percentage |
| :--- | :--- | :--- |
{orig_table}

### 3.2 Normalized TruthLens Labels
| Canonical Label | Count | Percentage |
| :--- | :--- | :--- |
{norm_table}

---

## 4. Modality Availability Breakdown
| Modality Feature | Available Records | Availability Ratio |
| :--- | :--- | :--- |
| **Text Available** | {ms.text_available:,} | {text_pct:.2f}% |
| **Image Available** | {ms.image_available:,} | {img_pct:.2f}% |
| **Multimodal (Text + Image)** | {ms.text_and_image_available:,} | {both_pct:.2f}% |
| **Metadata Available** | {ms.metadata_available:,} | {meta_pct:.2f}% |
| **Timestamp Available** | {ms.timestamp_available:,} | {ts_pct:.2f}% |
| **Source / Domain Available** | {ms.source_available:,} | {src_pct:.2f}% |

---

## 5. Deduplication & Leakage Defense
* **Unique Content Hashes (SHA-256)**: {ds.unique_content_hashes:,}
* **Exact Duplicate Records**: {ds.exact_duplicates_count:,}
* **Exact Duplicate Clusters**: {ds.exact_duplicate_groups:,}
* **Near-Duplicate Clusters (MinHash / Jaccard)**: {ds.near_duplicate_groups:,}
* **Leakage Defense Policy**: All records belonging to identical or near-duplicate clusters are constrained to the same split.

---

## 6. Partitioning & Split Statistics
| Split Partition | Record Count | Proportion |
| :--- | :--- | :--- |
| **Train Split** | {ss.train_count:,} | {train_pct:.2f}% |
| **Validation Split** | {ss.val_count:,} | {val_pct:.2f}% |
| **Test Split** | {ss.test_count:,} | {test_pct:.2f}% |

### 6.1 Split Label Distribution
* **Train**: `{dict(ss.train_distribution)}`
* **Validation**: `{dict(ss.val_distribution)}`
* **Test**: `{dict(ss.test_distribution)}`
{reassign_sec}
---

## 7. Quality & Research Integrity Statement
All statistics documented in this report are computed directly from the processed dataset artifact. No metrics, missingness ratios, or class counts have been estimated or fabricated.
"""
        return report

    @classmethod
    def save_report(cls, manifest: DatasetManifest, output_path: Path) -> Path:
        """Generates and writes Markdown quality report."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = cls.generate_markdown_report(manifest)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info("Saved dataset quality report to %s", output_path)
        return output_path
