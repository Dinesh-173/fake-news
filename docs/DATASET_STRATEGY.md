# TruthLens: Dataset Strategy

## 1. Candidate Datasets & Modality Breakdown

TruthLens evaluates and integrates multiple standardized academic datasets to guarantee experimental variety, cross-domain validation, and multimodal coverage:

| Dataset | Primary Modality | Domain / Source | Native Label Granularity | TruthLens Target Normalization |
| :--- | :--- | :--- | :--- | :--- |
| **LIAR** (Wang, 2017) | Text (short statements) | PolitiFact political statements | 6 classes (`pants-fire`, `false`, `barely-true`, `half-true`, `mostly-true`, `true`) | 4 classes: `false`/`pants-fire` -> Likely False; `barely-true`/`half-true` -> Misleading; `mostly-true`/`true` -> Likely True |
| **FakeNewsNet** (Shu et al., 2019) | Multimodal (Text + Images + Social context) | PolitiFact & GossipCop | Binary (`real`, `fake`) | 2 classes (binary baseline) or mapped to Likely True / Likely False |
| **Fakeddit** (Compute et al., 2020) | Multimodal (Submissions + Images) | Reddit diverse subreddits | 2-class, 3-class, and 6-class variants | 6-class or normalized to 4-tier TruthLens schema |
| **PHEME** (Zubiaga et al., 2016) | Text + Temporal Threads | Breaking news rumors on Twitter | 3 classes (`rumour`, `non-rumour`, stance labels) | Stance detection and temporal evaluation |

---

## 2. Explicit Label Mapping Protocols

Never silently alter or merge dataset labels. All mappings must be formally defined, deterministic, and recorded in configuration files.

### 2.1 PolitiFact / LIAR 6-Class Mapping Schema
```python
LIAR_TO_TRUTHLENS_MAP = {
    "pants-fire": "LIKELY_FALSE",
    "false": "LIKELY_FALSE",
    "barely-true": "MISLEADING_OR_PARTIALLY_SUPPORTED",
    "half-true": "MISLEADING_OR_PARTIALLY_SUPPORTED",
    "mostly-true": "LIKELY_TRUE",
    "true": "LIKELY_TRUE",
}
```

### 2.2 Binary Dataset Mapping (FakeNewsNet)
```python
BINARY_TO_TRUTHLENS_MAP = {
    "real": "LIKELY_TRUE",
    "fake": "LIKELY_FALSE"
}
```
*Note*: For binary datasets, the `MISLEADING` and `INSUFFICIENT_EVIDENCE` states are reserved for model outputs where evidence is conflicting or insufficient, preserving the distinction between source training labels and model operational states.

---

## 3. Modality Handling & Missing Data Strategy

1. **Text-Only Samples in Multimodal Benchmarks**: If an image is absent, pass a designated zero/null visual embedding along with an explicit boolean flag `has_image = False`. The multimodal fusion layer must be trained to handle missing image tokens via masked attention or zero-imputation without crashing.
2. **Missing Metadata (Author, Publisher, Timestamp)**: Impute missing categorical features with an explicit `"UNKNOWN"` token to prevent silent data drop or bias.
3. **Low-Resolution / Corrupted Images**: Filter out unparseable files during ingestion; mark image status as `CORRUPT` and fallback to unimodal text processing with an explicit warning in the audit log.
