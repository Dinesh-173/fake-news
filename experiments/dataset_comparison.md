# TruthLens: Candidate Benchmark Datasets Comparison Matrix

> **Research Policy**: TruthLens strictly reports empirical status. Unavailable datasets are explicitly
> marked as unacquired; no statistics are simulated or assumed.

| Research Dimension | LIAR (Wang, 2017) | FakeNewsNet (Shu, 2019) | Fakeddit (Nakamura, 2020) | NELA-GT (Gruppi, 2022) |
| :--- | :--- | :--- | :--- | :--- |
| **Acquisition Status** | **AVAILABLE LOCALLY** | *NOT ACQUIRED* | *NOT ACQUIRED* | *NOT ACQUIRED* |
| **Verified Records** | 12,836 (Train=10,269, Val=1,284, Test=1,283) | N/A | N/A | N/A |
| **Veracity Labels** | 6-class fine-grained veracity (pants-fire, false, barely-true, half-true, mostly-true, true) | Binary veracity (real, fake) across PolitiFact and GossipCop | Multi-task (2-way, 3-way, and 6-way fine-grained veracity labels) | Outlet-level veracity and bias scores from Media Bias/Fact Check |
| **Canonical Mapping** | 3-class TruthLens canonical (LIKELY_TRUE, LIKELY_FALSE, MISLEADING_OR_PARTIALLY_SUPPORTED) | 2-class TruthLens canonical (LIKELY_TRUE, LIKELY_FALSE) | TruthLens canonical mapped via data/label_mappings/fakeddit.json | Aggregated veracity tiers mapped via data/label_mappings/nela.json |
| **Modalities** | Textual statements (100% text, 0% local images) | Multimodal (Article text, web scrape HTML, tweet social context graphs) | Multimodal (Reddit submission title, comment text, and linked submission image) | Full-text news articles from mainstream and alternative outlets |
| **Temporal Metadata** | Not present in canonical TSV format (100% missing) | Tweet timestamps present in hydrated Twitter JSON | Unix epoch creation timestamp of Reddit submission | Article publication dates (2018–2022) |
| **Source Metadata** | Speaker, speaker job, state info, party affiliation, venue/context | News publisher URL and Twitter user profiles | Subreddit name, Reddit submission author, domain link | News outlet name, article URL |
| **Pipeline Status** | Phase 2 Pipeline Verified (Cleaned, Deduplicated, Leakage-Free Splitting) | Loader implemented; raw data acquisition pending | Loader implemented; raw data acquisition pending | Loader implemented; raw SQLite/JSON acquisition pending |

---

### Dataset Acquisition Directives

1. **LIAR**: Fully acquired, locally verified (12,836 statements across train, valid, test). Serves as the primary textual veracity benchmark for Phase 4 baselines.
2. **FakeNewsNet**: Requires social context hydration; recommended for Phase 7 (evidence retrieval) and graph propagation experiments.
3. **Fakeddit**: Required for Phase 9 (vision processing) and Phase 10 (multimodal fusion) due to paired image-text Reddit submissions.
4. **NELA-GT**: Recommended for cross-dataset transfer testing (Phase 13) to assess long-form article veracity generalizability.