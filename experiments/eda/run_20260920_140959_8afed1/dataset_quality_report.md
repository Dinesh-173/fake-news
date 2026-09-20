# TruthLens: Dataset Quality Analysis & Statistical Profile — LIAR

> **Research Note**: This document presents empirical statistical measurements computed directly from the
> benchmark dataset. All statistics reflect the acquired corpus and follow the TruthLens Phase 3 protocol.

---

## 1. Dataset Identity & Provenance

| Dimension | Value |
| :--- | :--- |
| **Dataset Name** | `liar` |
| **Dataset Version** | `v1.0.0 (ACL 2017)` |
| **Profile Run ID** | `run_20260920_140959_8afed1` |
| **Analysis Timestamp (UTC)** | `2026-09-20T14:10:00.238054+00:00` |
| **Total Processed Records** | `12,836` |
| **Git Commit** | `f64a5a6f3cfa799e76d3e292c0ccbf28af74bf89` |
| **Python Runtime** | `3.14.7` (Windows-11-10.0.26200-SP0) |
| **Random Seed** | `42` |

### Verified Input File Hashes (SHA-256)

| File Key | SHA-256 Digest |
| :--- | :--- |
| `train` | `23ef8cb7ff57b25c941100d0744039cebc595d65a67ea190f72b330a658d8c36` |
| `valid` | `8dd3d7ed1a11e78ed7df090c9eddb0f91147f895850c78d49dd7b4e88fd09a16` |
| `test` | `ee41e2c741e0019216952625145af8c4aa8c8d3057ec68284b134267bb845e6f` |

---

## 2. Label Distribution & Normalization Coverage

### 2.1 Original Dataset Labels

| Original Label | Count | Percentage |
| :--- | :--- | :--- |
| `half-true` | 2,638 | 20.55% |
| `false` | 2,511 | 19.56% |
| `mostly-true` | 2,466 | 19.21% |
| `barely-true` | 2,108 | 16.42% |
| `true` | 2,063 | 16.07% |
| `pants-fire` | 1,050 | 8.18% |

### 2.2 TruthLens Normalized Labels

| Normalized Canonical Label | Count | Percentage |
| :--- | :--- | :--- |
| **`MISLEADING_OR_PARTIALLY_SUPPORTED`** | 4,746 | 36.97% |
| **`LIKELY_TRUE`** | 4,529 | 35.28% |
| **`LIKELY_FALSE`** | 3,561 | 27.74% |

### 2.3 Label Mapping Provenance Breakdown

| Original Label | Normalized Target | Status | Count | Percentage |
| :--- | :--- | :--- | :--- | :--- |
| `half-true` | `MISLEADING_OR_PARTIALLY_SUPPORTED` | `mapped` | 2,638 | 20.55% |
| `false` | `LIKELY_FALSE` | `mapped` | 2,511 | 19.56% |
| `mostly-true` | `LIKELY_TRUE` | `mapped` | 2,466 | 19.21% |
| `barely-true` | `MISLEADING_OR_PARTIALLY_SUPPORTED` | `mapped` | 2,108 | 16.42% |
| `true` | `LIKELY_TRUE` | `mapped` | 2,063 | 16.07% |
| `pants-fire` | `LIKELY_FALSE` | `mapped` | 1,050 | 8.18% |

### 2.4 Class Balance & Entropy

- **Mapping Coverage**: 100.00% (0 unresolved records)
- **Majority Class**: `MISLEADING_OR_PARTIALLY_SUPPORTED` (36.97%)
- **Minority Class**: `LIKELY_FALSE` (27.74%)
- **Imbalance Ratio**: 1.33:1 (Majority to Minority)
- **Normalized Shannon Entropy**: 0.9932 (Scale: 0.0 = total monopoly, 1.0 = equal uniform distribution)

---

## 3. Text Statistics & Linguistic Quality Audit

**Tokenizer Employed**: `Regex-based Word Tokenizer (\b\w+(?:'\w+)?\b|[^\w\s])`

### 3.1 Length Distributions

| Metric | Character Count | Token Count |
| :--- | :--- | :--- |
| **Minimum** | 11.0 | 2.0 |
| **Maximum** | 395.0 | 88.0 |
| **Mean (μ)** | 106.41 | 21.06 |
| **Median (P50)** | 99.00 | 20.00 |
| **Std. Dev. (σ)**| 45.03 | 9.01 |
| **P5 (5th %tile)**| 47.0 | 9.0 |
| **P25 (1st Quartile)**| 73.0 | 14.0 |
| **P75 (3rd Quartile)**| 133.0 | 26.0 |
| **P95 (95th %tile)**| 191.0 | 38.0 |
| **P99 (99th %tile)**| 234.0 | 48.0 |

### 3.2 Class-Conditional Character Lengths

| Normalized Label | Count | Mean Chars | Median Chars | Std Dev | P5 | P95 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `LIKELY_FALSE` | 3,561 | 100.9 | 93.0 | 43.5 | 41.0 | 181.0 |
| `LIKELY_TRUE` | 4,529 | 106.7 | 99.0 | 44.1 | 50.0 | 189.0 |
| `MISLEADING_OR_PARTIALLY_SUPPORTED` | 4,746 | 110.2 | 103.0 | 46.5 | 48.0 | 196.75 |

### 3.3 Textual Anomaly & Quality Audit

| Quality Flag | Triggered Records | Percentage | Scientific Definition / Rationale |
| :--- | :--- | :--- | :--- |
| **Empty Content** | 0 | 0.00% | Statements lacking text cannot be verified. |
| **Whitespace Only** | 0 | 0.00% | Strings containing only whitespace characters. |
| **Extremely Short (<15 chars)** | 9 | 0.07% | Text too brief to encode verifiable factual propositions. |
| **Extremely Long (>5000 chars)** | 0 | 0.00% | Disproportionate outliers indicating multi-article concatenation. |
| **Excessive Punctuation (>25%)** | 2 | 0.02% | Punctuation density suggesting spam or encoding errors. |
| **Excessive URLs (>2 URLs)** | 0 | 0.00% | Link-heavy spam rather than self-contained claims. |
| **Repetitive Text (<50% unique)** | 3 | 0.02% | Low token vocabulary diversity indicative of bot text. |
| **Total Flagged Records** | **14** | **0.11%** | Unique records triggering at least one anomaly condition. |

---

## 4. Source & Publisher Distribution

- **Unique Sources / Speakers**: 5,162
- **Unique Domains Extracted**: 191
- **Top-5 Source Concentration Share**: 10.81%
- **Top-10 Source Concentration Share**: 16.77%
- **Herfindahl-Hirschman Index (HHI)**: 39.94 (Normalized: 0.0038)

### 4.1 Top-10 Sources / Speakers

| Rank | Source / Speaker | Count | Percentage |
| :--- | :--- | :--- | :--- |
| 1 | `a news release` | 318 | 2.48% |
| 2 | `an interview` | 287 | 2.24% |
| 3 | `a press release` | 286 | 2.23% |
| 4 | `a speech` | 267 | 2.08% |
| 5 | `a TV ad` | 229 | 1.78% |
| 6 | `a tweet` | 191 | 1.49% |
| 7 | `a campaign ad` | 165 | 1.29% |
| 8 | `a television ad` | 162 | 1.26% |
| 9 | `a radio interview` | 129 | 1.00% |
| 10 | `a debate` | 118 | 0.92% |

---

## 5. Temporal Coverage & Longitudinal Stability

- **Timestamps Available**: 0 (0.00%)
- **Timestamps Missing**: 12,836
- **Temporal Data Present**: NO (Dataset lacks explicit timestamp column)
- **Earliest Timestamp**: `N/A`
- **Latest Timestamp**: `N/A`

### Chronological Observations

- No valid publication timestamps detected in this dataset.
- Temporal distribution and drift analysis cannot be empirically evaluated.
- Temporal splits should not be applied to datasets lacking validated chronological metadata.

---

## 6. Metadata Completeness & Attribute Quality

- **Overall Mean Metadata Completeness**: 81.62%

| Field Name | Available | Missing | Missing % | Unique Values | Category Entropy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `author` | 12,836 | 0 | 0.0% | 3,318 | 0.8289 |
| `source_name_raw` | 12,836 | 0 | 0.0% | 5,163 | 0.8561 |
| `source_domain` | 244 | 12,592 | 98.1% | 190 | 0.9603 |
| `speaker` | 12,836 | 0 | 0.0% | 3,318 | 0.8289 |
| `party_affiliation` | 10,651 | 2,185 | 17.0% | 23 | 0.3382 |
| `context` | 12,707 | 129 | 1.0% | 5,079 | 0.8549 |
| `state_info` | 10,085 | 2,751 | 21.4% | 72 | 0.7199 |
| `subjects` | 12,836 | 0 | 0.0% | 4,547 | 0.8281 |
| `job_title` | 9,258 | 3,578 | 27.9% | 1,349 | 0.7264 |

---

## 7. Modality Availability

| Modality Combination | Record Count | Percentage |
| :--- | :--- | :--- |
| **Text Only** | 0 | 0.00% |
| **Image Only** | 0 | 0.00% |
| **Text + Image** | 0 | 0.00% |
| **Text + Metadata** | 12,836 | 100.00% |
| **Text + Image + Metadata** | 0 | 0.00% |

### Three-Tier Image Availability Verification

| Level | Status | Details |
| :--- | :--- | :--- |
| **Image Metadata Available** | `NO` | References/URLs present in record schemas. |
| **Image Binary Available** | `NO` | Local binary image files detected on filesystem. |
| **Image Analysis Feasible** | `NO` | Files successfully decoded by imaging subsystem. |

---

## 8. Duplicate & Near-Duplicate Analysis

- **Exact Duplicates Count**: 26 (0.20%)
- **Exact Duplicate Groups**: 24
- **Near-Duplicate Groups**: 1 (encompassing 2 records)
- **Conflicting Clusters (Label Divergence)**: 5

### Cluster Size Distribution

- Singletons: 12,786
- Pairs (Size 2): 22
- Moderate Clusters (Size 3-5): 2
- Large Clusters (>5): 0
- Maximum Cluster Size: 3

### Sample Clusters Exhibiting Veracity Divergence

| Cluster ID | Type | Records | Normalized Labels Present | Sample Statement Text |
| :--- | :--- | :--- | :--- | :--- |
| `exact_grp_57ce7ef3a00c` | `exact` | 2 | `LIKELY_FALSE, MISLEADING_OR_PARTIALLY_SUPPORTED` | On abortion... |
| `exact_grp_f067b94f8831` | `exact` | 2 | `LIKELY_FALSE, LIKELY_TRUE` | On support for gay marriage.... |
| `exact_grp_d61a4ac6faaa` | `exact` | 2 | `LIKELY_TRUE, MISLEADING_OR_PARTIALLY_SUPPORTED` | On torture.... |
| `exact_grp_2e0b5b86600e` | `exact` | 2 | `LIKELY_TRUE, MISLEADING_OR_PARTIALLY_SUPPORTED` | Four balanced budgets in a row, with no new taxes for anyone. The best job growth in 12 years. Nearly 130,000 new privat... |
| `exact_grp_365dbf1a3386` | `exact` | 2 | `LIKELY_FALSE, MISLEADING_OR_PARTIALLY_SUPPORTED` | On the status of illegal immigrants... |

---

## 9. Cross-Dataset & Cross-Split Overlap

- **Multi-Dataset Comparison**: `SINGLE DATASET LOADED`
- **Cross-Split Data Leakage**: `DETECTED (UNSAFE)`

- Evaluated pairwise contamination across 3 partitions (train, valid, test).
- CRITICAL WARNING: Split overlap detected! Data leakage is present between partitions.

---

## 10. Inferential Statistical Hypothesis Testing

The following statistical tests answer prespecified empirical questions regarding dataset properties.

### Test 1: Chi-Square Test of Independence (Party × Veracity)

- **Research Question**: *Is statement veracity independent of speaker political party affiliation?*
- **Null Hypothesis (H0)**: H0: Statement veracity and speaker political party affiliation are independent.
- **Alternative Hypothesis (H1)**: H1: Statement veracity and speaker political party affiliation are associated.
- **Sample Size (N)**: 10,651
- **Test Statistic**: Chi-Square (χ²) = `236.2576` (df=4)
- **p-value**: `5.93e-50` (Significance Level α = 0.05)
- **Statistical Significance**: `REJECT H0`
- **Effect Size**: Cramér's V = `0.1053`
- **Scientific Interpretation**: Statistically significant association detected (χ²=236.2576, df=4, p=5.93e-50, Cramér's V=0.1053). Observed statement truthfulness differs systematically across political affiliations in this benchmark. Note that Cramér's V of 0.1053 indicates the magnitude of association.
- **Key Assumptions & Limitations**:
  - Assumption: Contingency table expected frequencies >= 5 in at least 80% of cells.
  - Assumption: Mutual exclusivity of categorization.
  - Limitation: LIAR statements are fact-checked based on newsworthiness and editorial selection, not random sampling of all political speech.
  - Limitation: Statistical association reflects fact-checked claims in PolitiFact, not a generalized claim about political parties as a whole.

### Test 2: Mann-Whitney U Test (LIKELY_TRUE vs LIKELY_FALSE Text Length)

- **Research Question**: *Do statements categorized as LIKELY_TRUE and LIKELY_FALSE differ in character length?*
- **Null Hypothesis (H0)**: H0: Text length distributions for LIKELY_TRUE and LIKELY_FALSE are identical.
- **Alternative Hypothesis (H1)**: H1: Text length distributions for LIKELY_TRUE and LIKELY_FALSE differ.
- **Sample Size (N)**: 8,090
- **Test Statistic**: Mann-Whitney U = `8675418.5`
- **p-value**: `4.50e-09` (Significance Level α = 0.05)
- **Statistical Significance**: `REJECT H0`
- **Effect Size**: Rank-Biserial Correlation = `-0.0758`
- **Scientific Interpretation**: Statistically significant difference in character length between LIKELY_TRUE (median=99.0) and LIKELY_FALSE (median=93.0) statements (U=8675418.5, p=4.50e-09, Rank-biserial r=-0.0758). While statistically detectable, the effect size indicates practical length variation is modest.
- **Key Assumptions & Limitations**:
  - Assumption: Continuous or ordinal response variable (character length in characters).
  - Assumption: Non-parametric distribution (does not assume Gaussian normality).
  - Assumption: Independent observations across statements.
  - Limitation: Statistical significance in large sample sizes does not imply that text length alone is an effective predictive feature.
  - Limitation: Character length is sensitive to quotation conventions and speech transcript editing.

### Test 3: Chi-Square Goodness-of-Fit (Label Uniformity)

- **Research Question**: *Are statements uniformly distributed across truthfulness categories?*
- **Null Hypothesis (H0)**: H0: Statement records are uniformly distributed across all canonical labels.
- **Alternative Hypothesis (H1)**: H1: Statement records are not uniformly distributed across canonical labels.
- **Sample Size (N)**: 12,836
- **Test Statistic**: Chi-Square (χ²) = `186.0656` (df=2)
- **p-value**: `3.95e-41` (Significance Level α = 0.05)
- **Statistical Significance**: `REJECT H0`
- **Effect Size**: Cohen's w = `0.1204`
- **Scientific Interpretation**: Statistically significant deviation from uniform class distribution (χ²=186.0656, df=2, p=3.95e-41, Cohen's w=0.1204). The benchmark exhibits significant class disparity relative to an equal prior expectation.
- **Key Assumptions & Limitations**:
  - Assumption: Categories are mutually exclusive and exhaustive.
  - Assumption: Expected frequency per cell >= 5.
  - Limitation: Uniform distribution is tested as an agnostic statistical baseline null model; natural real-world veracity distributions are not expected to be uniform.
  - Limitation: Rejection of uniformity is standard for empirical media datasets and does not imply defective data collection.

---

## 11. Descriptive Research Implications

1. **Class Distribution**: The dataset exhibits non-uniform class distribution across categories. Classical and neural baseline models in Phase 4 must report macro-averaged F1 and class-conditional PR-AUC in addition to overall accuracy.
2. **Modality Constraints**: Binary image assets are not present in this text-based benchmark. Vision processing (Phase 9) and multimodal fusion (Phase 10) must be evaluated on multimodal candidate benchmarks (such as Fakeddit) once acquired.
3. **Speaker Concentration**: A substantial fraction of fact-checked claims originate from repeat political figures. Evaluators must distinguish between generalizable linguistic veracity signals and speaker identity memorization.
4. **Longitudinal Considerations**: Due to the absence of dedicated publication timestamps in the canonical TSV format, temporal chronological splits cannot be evaluated without external metadata enrichment.

---

## 12. Research Readiness Assessment

| Evaluation Dimension | Measurement & Empirical Status |
| :--- | :--- |
| **Label Completeness** | `100.0% (0 unresolved)` |
| **Modality Completeness** | `Text Only (No Local Images)` |
| **Temporal Completeness** | `0.0% Timestamps Present` |
| **Source Completeness** | `5162 Unique Sources (Top-5 Share: 10.81%)` |
| **Duplicate Risk** | `LOW (0.2% exact, 0.02% near)` |
| **Cross Dataset Leakage Risk** | `DETECTED` |
| **Class Balance** | `Imbalance Ratio: 1.33:1 (Normalized Entropy: 0.9932)` |
| **Reproducibility** | `Git: f64a5a6f | Python: 3.14.7 | Seed: 42` |

> [!IMPORTANT]
> **Research Readiness Verdict**: In accordance with the TruthLens engineering protocol, this assessment
> does not compute a single subjective composite score or rank datasets arbitrarily. The empirical evidence
> above confirms that textual, label, source, and duplicate dimensions are fully quantified and ready to
> establish evaluation baselines for Phase 4.
