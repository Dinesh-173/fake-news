# TruthLens: Empirical Ablation Study Results

| Configuration | Description | Image | Evidence | Source | Meta | Accuracy | Mean Confidence | Mean Sufficiency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `full_model` | Full Multimodal Evidence-Aware Framework (Text + Image + Evidence + Source + Metadata) | ✓ | ✓ | ✓ | ✓ | **0.8000** | 0.6602 | 0.3347 |
| `text_only` | Text Classifier Only (No external evidence, no image, no source credibility) | ✗ | ✗ | ✗ | ✗ | **0.4000** | 0.4316 | 0.0000 |
| `text_plus_image` | Text + Image (No external retrieval, no source credibility) | ✓ | ✗ | ✗ | ✗ | **0.6000** | 0.4316 | 0.0000 |
| `text_plus_evidence` | Text + Evidence (Unweighted external evidence, no image, no source credibility) | ✗ | ✓ | ✗ | ✗ | **0.4000** | 0.4114 | 0.1775 |
| `text_plus_source_meta` | Text + Source/Metadata (No image, no retrieved evidence passages) | ✗ | ✗ | ✓ | ✓ | **0.4000** | 0.4316 | 0.0000 |
| `minus_image` | Ablation: Full Framework excluding Image Analysis | ✗ | ✓ | ✓ | ✓ | **0.8000** | 0.6602 | 0.3347 |
| `minus_evidence` | Ablation: Full Framework excluding Evidence Retrieval | ✓ | ✗ | ✓ | ✓ | **0.6000** | 0.4316 | 0.0000 |
| `minus_source` | Ablation: Full Framework excluding Source Credibility | ✓ | ✓ | ✗ | ✓ | **0.4000** | 0.4114 | 0.1775 |
| `minus_metadata` | Ablation: Full Framework excluding Claim Metadata & Salience | ✓ | ✓ | ✓ | ✗ | **0.8000** | 0.6602 | 0.3347 |
