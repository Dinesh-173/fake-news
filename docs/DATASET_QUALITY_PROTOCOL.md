# TruthLens: Dataset Quality Protocol

## 1. Quality Assurance Pipeline

Every raw dataset ingested into TruthLens must traverse a four-stage validation pipeline:

```text
  Raw Dataset File (CSV/JSONL/Parquet)
                   │
                   ▼
       [Stage 1: Structural Schema Validation]
         - Verify mandatory fields (id, text, label, split)
         - Enforce data types and non-empty constraints
                   │
                   ▼
       [Stage 2: Cleaning & Normalization]
         - Strip invalid unicode, HTML tags, null bytes
         - Standardize whitespace and casing
                   │
                   ▼
       [Stage 3: Deduplication & Leakage Check]
         - Exact hash deduplication (SHA-256)
         - Near-duplicate detection (MinHash / Jaccard similarity)
         - Cross-split contamination audit
                   │
                   ▼
       [Stage 4: Quality Reporting & Profiling]
         - Class distribution analysis
         - Text length percentiles (p50, p90, p99)
         - Missing modality ratio
         - Versioned Parquet artifact emission
```

---

## 2. Leakage Defense Standards

1. **Exact Duplicate Leakage**: Articles with identical textual bodies or headlines must never appear in both training and evaluation splits.
2. **Near-Duplicate Leakage**: News wires frequently reprint syndicated stories with minor editorial edits. TruthLens enforces a MinHash LSH filter with Jaccard threshold $\theta = 0.85$ to ensure syndicated wire copies are relegated strictly to the same split.
3. **Temporal Contamination**: Under temporal evaluation protocols, no training sample may possess a publication timestamp $t_{\text{train}} > \min(t_{\text{val}}, t_{\text{test}})$.
4. **Source Over-representation**: Source domain distributions must be profiled to avoid models relying on publisher identity as a shortcut feature (e.g., memorizing that domain $X$ is always labeled "fake").

---

## 3. Data Versioning & Manifest Specifications

Every processed dataset artifact is assigned a semantic version (`v1.0`, `v1.1`, etc.) and accompanied by a cryptographic manifest (`manifest.json`):

```json
{
  "dataset_name": "liar_normalized",
  "dataset_version": "v1.0.0",
  "source_url": "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip",
  "created_at": "2026-09-20T12:00:00Z",
  "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "total_records": 12791,
  "splits": {
    "train": 10240,
    "val": 1284,
    "test": 1267
  },
  "class_distribution": {
    "LIKELY_TRUE": 3820,
    "LIKELY_FALSE": 4510,
    "MISLEADING_OR_PARTIALLY_SUPPORTED": 4461
  },
  "missing_modality_rate": {
    "images": 1.0,
    "metadata_speaker": 0.02
  }
}
```
*Note*: No numbers in any manifest may be fabricated; all manifests are generated programmatically directly from the loaded dataframe.
