# TruthLens: Project Specification

## 1. Executive Overview

**Project Title**: TruthLens: An Evidence-Aware Multimodal AI Framework for Fake News Detection and Explainable News Verification  
**Target Level**: University-level academic research and production-grade software prototype  
**Lead Architectural Paradigm**: Modular, Evidence-Grounded, Multimodal, Uncertainty-Aware Machine Learning System  

Modern online misinformation is characterized by rapid diffusion, multi-faceted presentation (multimodal text and manipulated or out-of-context imagery), and subtle semantic distortions. Conventional deep learning approaches typically classify articles as binary labels ("Real" vs. "Fake") in an opaque end-to-end manner, or query Large Language Models (LLMs) as black-box arbiters. Both paradigms suffer from severe limitations:
1. Susceptibility to hallucinations and lack of verifiable provenance.
2. Inability to distinguish between low-confidence predictions and missing evidence.
3. Rapid degradation when exposed to out-of-distribution or temporal drift.

TruthLens addresses these foundational challenges by decoupling claim extraction, task-specific deep modeling, real-time external evidence retrieval, cross-modal verification, and calibrated uncertainty quantification, using generative LLMs solely for structured explanation and synthesis rather than unsubstantiated factual judgment.

---

## 2. Core Research Objectives

1. **Evidence-Grounded Verification**: Ensure every verification decision is directly supported by traceable, timestamped, external evidence items with quantified stance relationships (Supports, Contradicts, Contextualizes, or Unknown).
2. **Multimodal Coherence & Stance Alignment**: Jointly model textual claims and visual imagery to identify cross-modal inconsistencies (e.g., mismatched captions, cheap fakes, recycled contextual imagery) and OCR textual overlaps.
3. **Calibrated Confidence & Explicit Uncertainty**: Explicitly decouple model prediction confidence from evidence sufficiency, exposing when the framework lacks sufficient external facts (`INSUFFICIENT EVIDENCE`) rather than manufacturing high-confidence falsehoods.
4. **Guarded Explainability**: Produce human-interpretable natural language rationales constrained exclusively to retrieved factual snippets and model intermediate activations, with strict architectural defenses against prompt injection and hallucination.
5. **Reproducible Empirical Benchmarks**: Provide deterministic, automated pipelines for cross-dataset, temporal, and ablation evaluations on standardized datasets without fabricated metrics.

---

## 3. Credibility State Taxonomies

To avoid forced binary oversimplification, TruthLens standardizes on a four-tier credibility classification:

| Credibility State | Definition | Operational Criteria |
| :--- | :--- | :--- |
| **LIKELY TRUE** | The factual claims are corroborated by high-authority external sources with negligible contradiction and consistent cross-modal context. | High agreement among retrieved evidence; stance alignment >= 0.75; model confidence high; evidence sufficiency >= 0.70. |
| **LIKELY FALSE** | The core claims directly contradict verified factual reporting, consensus datasets, or exhibit fabricated/tampered modal context. | Strong contradicting evidence from established sources; stance contradiction score >= 0.70; evidence sufficiency >= 0.70. |
| **MISLEADING / PARTIALLY SUPPORTED** | The article contains true factual elements embedded within false framing, cherry-picked context, or non-matching imagery. | Conflicting stance scores; high cross-modal semantic distance; key claims unsubstantiated despite partial factual overlap. |
| **INSUFFICIENT EVIDENCE** | External retrieval returns low-relevance snippets, disputed claims without consensus, or search failures. | Evidence sufficiency < 0.40 or retrieval failure. System explicitly halts assertive classification to prevent false certainty. |

---

## 4. System Boundaries & Explicit Limitations

### 4.1 What TruthLens Is
- An automated decision-support and explainability framework for journalists, researchers, and critical news readers.
- An experimental testbed evaluating multimodal fusion, evidence retrieval efficacy, and calibration techniques.
- A provenance-preserving pipeline where every conclusion links to verifiable primary sources.

### 4.2 What TruthLens Is NOT
- **Not an absolute arbiter of subjective truth**: Opinions, editorials, satire without factual claims, and religious or philosophical debates fall outside the verification scope.
- **Not an autonomous censorship engine**: TruthLens presents evidence, stances, and confidence breakdowns; it does not block or ban content.
- **Not a black-box LLM classifier**: An LLM is never prompted with "Is this article true or fake?" to produce the final decision.
