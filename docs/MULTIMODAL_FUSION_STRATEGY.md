# TruthLens: Multimodal Fusion Strategy

## 1. Modality Representations

TruthLens processes heterogeneous inputs across three primary representation streams:

1. **Text Representation ($\mathbf{h}_{\text{text}} \in \mathbb{R}^{d_t}$)**: Extracted from fine-tuned Transformer encoder (e.g., `roberta-base`, $d_t = 768$) pooled over claims and article body.
2. **Vision Representation ($\mathbf{h}_{\text{vis}} \in \mathbb{R}^{d_v}$)**: Extracted from Vision Transformer (e.g., `google/vit-base-patch16-224`, $d_v = 768$) applied to article images.
3. **Evidence Stance Representation ($\mathbf{h}_{\text{evid}} \in \mathbb{R}^{d_e}$)**: Dense vector summarizing top-$K$ evidence stance distributions, relevance scores, source authority priors, and conflict metrics ($d_e = 128$).
4. **Cross-Modal Consistency Representation ($\mathbf{h}_{\text{cross}} \in \mathbb{R}^{d_c}$)**: Derived from projected text-image cosine distance and OCR text similarity ($d_c = 64$).

---

## 2. Progressive Fusion Architecture

```text
    Text Vector       Vision Vector      Evidence Vector    Cross-Modal Vector
    (h_text)           (h_vis)            (h_evid)           (h_cross)
       │                  │                  │                  │
       ▼                  ▼                  ▼                  ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Linear Layer │   │ Linear Layer │   │ Linear Layer │   │ Linear Layer │
│  (Projection)│   │  (Projection)│   │  (Projection)│   │  (Projection)│
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │                  │
       └──────────────────┴─────────┬────────┴──────────────────┘
                                    │
                                    ▼
                         [Cross-Modal Attention Layer]
                         - Query: Claim Representation
                         - Key/Value: Evidence & Vision Tokens
                                    │
                                    ▼
                         [Gated Concatenation & LayerNorm]
                                    │
                                    ▼
                         [Multilayer Perceptron (MLP)]
                         - Linear(d_fused, 256) -> GELU -> Dropout(0.2)
                         - Linear(256, num_classes)
                                    │
                                    ▼
                             Raw Logits (z)
```

### 2.1 Baseline: Feature Concatenation
The baseline fusion model concatenates projected embeddings:
$$\mathbf{h}_{\text{fused}} = [\mathbf{W}_t \mathbf{h}_{\text{text}} \,\|\, \mathbf{W}_v \mathbf{h}_{\text{vis}} \,\|\, \mathbf{W}_e \mathbf{h}_{\text{evid}} \,\|\, \mathbf{W}_c \mathbf{h}_{\text{cross}}]$$
Passed through a 2-layer MLP classifier with ReLU/GELU activations.

### 2.2 Advanced: Gated Cross-Attention
When an image is absent, the gating network down-weights the visual branch to zero:
$$\mathbf{g} = \sigma(\mathbf{W}_g \mathbf{h}_{\text{text}} + b_g)$$
$$\mathbf{h}_{\text{fused}} = \mathbf{h}_{\text{text}} + \mathbf{g} \odot \text{CrossAttn}(\mathbf{h}_{\text{text}}, \mathbf{h}_{\text{vis}}) + \text{EvidenceAttn}(\mathbf{h}_{\text{text}}, \mathbf{h}_{\text{evid}})$$
This guarantees graceful unimodal degradation without architectural branching or runtime exceptions.
