# NarrativeCast

Predicting Emerging Narratives in Social Media Using Temporal Semantic Representations  
IE403 - Social Media Mining| University of Information Technology (UIT - VNU-HCM)

---

## Goal

NarrativeCast aims to investigate whether temporal semantic representations of social-media discussions can help predict which narrative will become dominant in a future time window.

---

## Data & External Artifacts

All large datasets and precomputed artifacts are hosted on Kaggle:

| Resource | Format / Size | Kaggle Link | Local Path | Description |
| :--- | :--- | :---: | :--- | :--- |
| **Raw Dataset** | `.parquet`  | [narrativecastdataset-ie403](https://www.kaggle.com/datasets/pgnjams/narrativecastdataset-ie403) | `data/raw/reddit_technology_madoc-001.parquet` | Original uncleaned reddit technology dataset (2014-2020) |
| **Clean Dataset** | `.parquet` | [narrativecast-clean-ie403](https://www.kaggle.com/datasets/pgnjams/narrativecast-clean-ie403) | `data/processed/reddit_technology_clean.parquet` | Cleaned reddit dataset and helper modules (`encode_posts.py`, `embedding_store.py`) |
| **Discovery Artifacts** | `.npz` (~191 MB), `.parquet` (~7.5 MB) | [narrativecast-narrative-discovery-artifacts](https://www.kaggle.com/datasets/pgnjams/narrativecast-narrative-discovery-artifacts) | `outputs/embeddings/` | Precomputed SBERT embeddings (`bigtech_antitrust_2017.npz`) and cluster table (`narrative_clusters_v1.parquet`) |

---

## Directory Structure

```text
NarrativeCast/
├── data/
│   ├── processed/
│   ├── raw/
│   └── sample/
├── docs/
├── notebooks/
│   ├── 01_baseline_feasibility.ipynb
│   ├── 01_data_profiling.ipynb
│   ├── 02_temporal_eda.ipynb
│   ├── 03_content_subdomain_eda.ipynb
│   └── 04_narrative_discovery.ipynb
├── outputs/
│   ├── content_eda/
│   ├── embeddings/
│   ├── narrative_discovery/
│   ├── reports/
│   └── temporal_eda/
├── reports/
│   └── results/
├── scripts/
│   ├── analysis/
│   └── preprocessing/
├── src/
│   └── embeddings/
└── README.md
```
## Current Phase

Dataset research and inspection.