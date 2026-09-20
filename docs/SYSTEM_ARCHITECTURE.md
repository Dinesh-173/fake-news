# TruthLens: System Architecture

## 1. High-Level Dataflow Pipeline

```text
               Raw Input (Text, URL, Headline, Image)
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │     Preprocessing     │
                     │  & Input Normalization│
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │    Claim Extraction   │
                     │  (Span & Salience)    │
                     └───────────┬───────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
┌──────────────────┐    ┌──────────────────┐   ┌───────────────────┐
│  Text Analysis   │    │  Image Analysis  │   │Evidence Retrieval │
│(RoBERTa/DeBERTa) │    │(OCR + Vision-Enc)│   │  & Query Engine   │
└────────┬─────────┘    └────────┬─────────┘   └─────────┬─────────┘
         │                       │                       │
         │                       ▼                       ▼
         │             ┌──────────────────┐    ┌───────────────────┐
         │             │Cross-Modal Check │    │ Stance & Relevance│
         │             │(Semantic Match)  │    │  (NLI Inference)  │
         │             └─────────┬────────┘    └─────────┬─────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ Multimodal & Evidence │
                     │   Fusion Classifier   │
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │Uncertainty Estimation │
                     │ & Temperature Scaling │
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ Structured Assessment │
                     │  (State + Provenance) │
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │  Guarded LLM Explainer│
                     │ (Evidence-Constrained)│
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │  API Response / UI    │
                     │ (Full Traceability)   │
                     └───────────────────────┘
```

---

## 2. Component Decomposition

### 2.1 Preprocessing & Ingestion Service
- **Input Types**: Plain text, news article URL (using readability parsers), raw headline, uploaded image (JPEG/PNG/WebP).
- **Validation**: Strict boundary checks (max characters: 25,000; min characters: 20; max image size: 10MB).
- **Sanitization**: Strip non-printable ASCII/unicode control characters; normalize whitespace; preserve punctuation markers critical for claim parsing.

### 2.2 Claim Extraction Engine
- Extracts discrete, testable factual propositions ($c_1, c_2, \dots, c_k$) from the continuous text.
- Retains exact character offset spans (`start_char`, `end_char`) and sentence index within original body for zero-loss traceability.
- Computes claim salience to rank assertions by factual criticality rather than trivial rhetorical statements.

### 2.3 Evidence Retrieval Engine
- **Query Generation**: Transforms extracted claims into targeted keyword and semantic search queries.
- **Multi-Source Retrieval**: Queries local fact-checking index (pgvector), pre-indexed academic/reputable corpora, and dynamic external search APIs (DuckDuckGo Search, scholarly repositories).
- **Candidate Reranking**: Utilizes dense bi-encoders (`all-MiniLM-L6-v2` or `bge-small-en-v1.5`) to score semantic similarity against candidate snippets.
- **Provenance Retention**: Each evidence snippet stores source domain, author/publisher, exact URL, publication timestamp, retrieval timestamp, and raw text chunk.

### 2.4 Evidence Analysis & Stance Detection
- Pairs claim $c_i$ with evidence snippet $e_j$.
- Evaluates Natural Language Inference (NLI) relationship:
  $$\text{Stance}(c_i, e_j) \in \{\text{SUPPORTS}, \text{CONTRADICTS}, \text{NEUTRAL / CONTEXTUALIZES}\}$$
- Derives aggregate evidence support score $S_{\text{supp}}$, contradiction score $S_{\text{contra}}$, and overall evidence sufficiency score:
  $$\text{Sufficiency} = \min\left(1.0, \frac{\sum_j \text{Relevance}(e_j) \cdot \mathbb{I}(\text{Stance} \ne \text{NEUTRAL})}{K_{\text{threshold}}}\right)$$

### 2.5 Vision & Cross-Modal Consistency Engine
- **Optical Character Recognition (OCR)**: Extracts embedded text from infographical or screenshot images using Tesseract OCR.
- **Vision Representation**: Generates visual embeddings using Vision Transformer (ViT) or ResNet backbones.
- **Cross-Modal Consistency**: Computes cosine similarity between textual claim embedding and image embedding space (e.g. via CLIP projection or multimodal joint space) to flag out-of-context image reuse.

### 2.6 Evidence-Aware Multimodal Fusion
- Concatenates text representations $\mathbf{h}_{\text{text}}$, vision representations $\mathbf{h}_{\text{vis}}$, and aggregated evidence stance vectors $\mathbf{h}_{\text{evid}}$.
- Feeds fused feature representations into calibrated dense layers with dropout and layer normalization.
- Employs residual connections to prevent dominance of one modality over weak signals from another.

### 2.7 Uncertainty & Calibration Module
- Separates classification logits $z_k$ into calibrated probabilities using temperature scaling parameter $T$:
  $$\hat{p}_k = \frac{e^{z_k / T}}{\sum_j e^{z_j / T}}$$
- Quantifies epistemic and aleatoric uncertainty.
- Enforces degradation gates: if evidence sufficiency falls below defined threshold $\tau_{\text{suff}}$, the system forces the output state to `INSUFFICIENT EVIDENCE`, suppressing fabricated confidence.

### 2.8 Guarded LLM Explanation Service
- Formats structured assessment payload containing:
  - Extracted claims and their offsets.
  - Retrieved evidence items with verified provenance and stance classifications.
  - Computed credibility state and uncertainty metrics.
- Prompts LLM using strict XML-encapsulated boundaries and explicit instructions:
  - "Explain the assessment using ONLY the verified evidence snippets provided. If an assertion is not in the snippets, do not state it as fact. Do not introduce external citations."
