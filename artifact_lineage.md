# Artifact Lineage & Provenance Map

This document establishes the exact end-to-end lineage for every table, figure, and numerical finding presented in the manuscript *Model–Partition Alignment in Federated Recommender Systems: A Methodological Evaluation of Horizontal and Item-Partitioned Learning* (INFORMATICA, 2026).

---

## 1. Manuscript Figures Lineage

| Figure | Title / Content | Generating Script | Input Data / Artifact | Output Image File | Output Verification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Figure 1** | Conceptual Alignment Taxonomy (HFL, IP-CF, Canonical VFL, Pseudo-VFL) | `experiments/generate_figures.py` (`create_conceptual_figure()`) | Analytical architectural specifications | `figures/conceptual-alignment.png` | Resolution: 300 DPI (3900×2550 px). Vectors/schematics rendered via matplotlib patches. |
| **Figure 2** | Round-Level MSE Trajectories & Round 9 Step Decay | `experiments/generate_figures.py` (`create_trajectories_figure()`) | Logged per-round MSE from pilot FL runs | `figures/round-trajectories.png` | Resolution: 300 DPI. Step decay annotation at round 9 ($0.2\eta$) aligned with Flower `on_fit_config_fn`. |
| **Figure 3** | Final MSE Point Estimates (Evaluation Units Separated) | `experiments/generate_figures.py` (`create_final_mse_figure()`) | `reproducibility_benchmark/experiments/ears_pilot_verified_results.json` | `figures/final-mse.png` | Resolution: 300 DPI. Panel 1: CF Macro (1.0076, 1.0458 for HFL; 1.0122, 1.0521 for IP-CF). Panels 2 & 3: CBF HFL Macro vs. VL-CBF Joint. |
| **Figure 4** | MovieLens 100K Multi-Seed Benchmark ($K \in \{2, 5, 10\}$) | `experiments/generate_figures.py` (`create_movielens_benchmark_figure()`) | `reproducibility_benchmark/experiments/benchmark_results.json` | `figures/movielens-benchmark.png` | Resolution: 300 DPI. 3-panel display: (a) Macro MSE error bars; (b) Recall@10 scaling; (c) Recall@10 CV dispersion. |

---

## 2. Manuscript Tables Lineage

| Table | Content | Generating Script / Source | Input Data | Corresponding Section | Status & Verification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Table 1** | Qualitative Comparison with Prior Work | Synthesized from literature audit | Citations / formal properties | Section 1 (Introduction) | Complete. Contrasts taxonomy, partition type, formal attenuation, ranking metrics, and client dispersion. |
| **Table 2** | Hyperparameter Transparency Specification | Audited from `server_app.py`, `client_app.py`, `models/neural_cf.py` | Implementation source files | Section 3.2 (Methodology) | Complete. Discloses embedding dimension ($d=32$), MLP structure (64$\to$32$\to$1), learning rate schedule, local epochs ($E=1$), and rounds ($T=10$). |
| **Table 3** | CF Final Evaluations on EARS Pilot Dataset | `reproducibility_benchmark/experiments/benchmark_pilot_ears.py` | `reproducibility_benchmark/data/ears/fl_datasets/` | Section 4.1 (Results) | Verified. All 14 client rows strictly satisfy $\text{RMSE} = \sqrt{\text{MSE}}$ with zero arithmetic discrepancy. |
| **Table 4** | CBF Final Evaluations (Macro vs. Joint) | Audited from historical content-based Flower logs | Feature-partitioned CBF logs | Section 4.1 (Results) | Discloses unit disparity (HFL client-level macro vs. VL-CBF centralized joint evaluation). |
| **Table 5** | MovieLens 100K Multi-Seed Benchmark ($K \in \{2, 5, 10\}$) | `reproducibility_benchmark/experiments/benchmark_movielens.py` | `reproducibility_benchmark/data/ml-100k/u.data` | Section 4.2 (Results) | Computed across 5 seeds [42, 43, 44, 45, 46]. Reports Macro MSE, RMSE, MAE, Recall@10, and NDCG@10 (Mean $\pm$ SD). |
| **Table 6** | Between-Client Dispersion on MovieLens 100K | `reproducibility_benchmark/experiments/benchmark_movielens.py` | `reproducibility_benchmark/data/ml-100k/u.data` | Section 4.2 (Results) | Computed across 5 seeds. Reports Client SD ($s_m$), CV ($s/\mu$), and Range for MSE and Recall@10. |

---

## 3. Dataset Preprocessing Lineage

| Dataset | Raw Source | Preprocessing Script | Clean Artifact | Matrix Statistics |
| :--- | :--- | :--- | :--- | :--- |
| **EARS Pilot** | `data/ears/*.csv` (orders, wishlist, views, client_data, product_list) | `reproducibility_benchmark/data/ears/preprocess_ears.py` | `data/ears/fl_datasets/*.csv` | 100 users, 25 items, 825 observed interactions ($33.0\%$ density), 1,675 unobserved pairs excluded from training/regression. |
| **MovieLens 100K** | GroupLens MovieLens 100K (`u.data`) | Stratified 80/20 train/test split per user (`split_train_test()`) | Dynamic train/test DataFrames per random seed | 943 users, 1,682 movies, 100,000 ratings (density = $6.30\%$). |

---

## 4. Test Suite Lineage

| Test File | Covered Modules | Pass Rate | Critical Properties Verified |
| :--- | :--- | :--- | :--- |
| `tests/test_validation_suite.py` | `federated_utils.py`, `ac_ipcf_trainer.py`, `ears_pilot_verified_results.json` | 100% (5/5 suites) | 1. Recall@10 & NDCG@10 exact math on edge ranks (1, 10, >10).<br>2. Candidate sampling strictly excludes historical train & test interactions.<br>3. Pointwise metric math ($\text{RMSE} = \sqrt{\text{MSE}}$) and dispersion calculation.<br>4. EARS verified results exactness.<br>5. FedAvg $1/K$ attenuation vs. AC-FedAvg active-coordinate recovery. |
