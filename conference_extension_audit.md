# Conference Extension Audit: Verification of the 30% New Content Requirement

**Target Journal:** *INFORMATICA* (Vilnius University / IOS Press)  
**Special Invitation:** *Invitation to publish papers from WorldCIST'26*  
**Journal Manuscript:** *Model–Partition Alignment in Federated Recommender Systems: A Methodological Evaluation of Horizontal and Item-Partitioned Learning*  
**Conference Paper of Origin:** Gaspar et al. (2027), *Optimizing collaborative filtering in federated learning by aligning the model with different data partitions*, In: *Recent Trends and Challenges in Information Systems and Technologies: WorldCIST 2026*, vol. 3, pp. 16--26. Springer, Cham. DOI: 10.1007/978-3-032-32026-1_2.  
**Auditor / Lead Author:** Tiago Pereira  
**Audit Date:** October 13, 2026  

---

## 1. Executive Summary & Policy Compliance

Journal special issues for post-conference extensions (*INFORMATICA*, IEEE, and Springer guidelines) require that an extended journal manuscript contain **at least 30% substantially new scientific and methodological contributions** beyond the original conference publication.

**Audit Conclusion:** The submitted journal manuscript exceeds the 30% requirement by a wide margin:
* **Textual Volume Increase:** **+130.6%** (~7,841 body words in the journal paper versus ~3,400 body words in the 11-page WorldCIST'26 proceedings).
* **Scientific Expansion:** Includes a full mathematical proof (Proposition 1), an active-coordinate aggregation mechanism (AC-FedAvg), an independent 5-seed benchmark on MovieLens 100K across $K \in \{2, 5, 10\}$ with ranking metrics (Recall@10, NDCG@10) and client-level dispersion analysis (SD, CV, Range), and an open-source reproducibility repository on GitHub.
* **Formal Citation & Title:** The conference paper is explicitly cited in Section 1 and in the References; the journal paper has an entirely new, methodological title.

---

## 2. Quantitative Comparison of Textual Volume

| Metric | WorldCIST'26 Conference Paper | INFORMATICA 2026 Journal Manuscript | Delta / Increase |
| :--- | :--- | :--- | :--- |
| **Page Format & Limit** | Springer LNCS (11 pages max) | INFORMATICA two-column format | N/A (different layout) |
| **Body Word Count** *(excl. references & headers)* | ~3,400 words | **7,841 words** | **+4,441 words (+130.6%)** |
| **Figures** | 2 small error plots | **4 high-resolution (300 DPI) multi-panel figures** | +2 figures (+100%) |
| **Tables** | 2 tables (CF and CBF MSE only) | **6 comprehensive tables** (incl. literature taxonomy, hyperparameters, multi-seed rankings, dispersion) | +4 tables (+200%) |
| **Mathematical Equations** | 2 basic equations | **16 numbered equations & formal proposition** | +14 equations |
| **Referenced Works** | 12 references | **29 comprehensive references with verified DOIs** | +17 references (+141%) |

---

## 3. Section-by-Section Lineage and Novelty Analysis

### Section 1: Introduction & Literature Taxonomy
* **Conference Paper Content:** Motivated federated learning and collaborative filtering at a high level; reported preliminary observation of item partition vs. horizontal partition on the pilot dataset.
* **Journal Novelty:**
  1. Formulated an explicit **Research Gap** prior to the contribution listing.
  2. Synthesized **Table 1** (qualitative literature taxonomy contrasting prior works across partition topology, aggregation mechanism, formal attenuation theory, ranking metrics, and client dispersion).
  3. Formulated 6 numbered, verifiable scientific contributions.
  4. Fully delineated the scope (clarifying utility and aggregation mechanics versus cryptographic privacy guarantees).

### Section 2: Federated Partitions, Update Attenuation, and Conceptual Taxonomy
* **Conference Paper Content:** Informal intuition about parameters being partitioned across clients.
* **Journal Novelty (100% NEW):**
  1. **Proposition 1 Formal Derivation:** Complete mathematical proof proving that dense sample-weighted FedAvg attenuates active coordinate embeddings by $p_k = n_k / \sum_\ell n_\ell$.
  2. **Corollary 1:** Balanced federation $1/K$ attenuation corollary.
  3. **Empirical Modulation Analysis:** Measured empirical $p_k$ long-tail distributions on MovieLens ($p_k \in [0.08, 0.51]$).
  4. **AC-FedAvg Formulation:** Defined Active-Coordinate Renormalized FedAvg ($\Delta \boldsymbol{q}_i^{\text{AC}}$) in Eq. 8 to isolate coordinate dilution from downstream MLP drift.
  5. **Figure 1:** Conceptual Alignment Taxonomy distinguishing HFL, coordinate-sliced partitioning, canonical VFL with split encoders, and pseudo-VFL zero-padding.

### Section 3: Experimental Methodology & Protocol Integrity
* **Conference Paper Content:** Brief informal description of Flower setup and EARS dataset.
* **Journal Novelty:**
  1. **Table 2:** Exhaustive hyperparameter transparency table.
  2. **EARS Ground Truth Audit:** Exact matrix statistics (100 users, 25 items, 825 observed pairs, 1,675 unobserved pairs).
  3. **Ranking Protocol Formalization:** Standardized leave-one-positive-out test protocol with 99 unobserved candidate items and deterministic RNG keying (`seed * 10000 + user_id`).
  4. **Epistemological Dispersion Distinction:** Differentiated user-cohort variance (fairness) in HFL from catalogue-tier variance (popularity equity) in IP-CF.

### Section 4: Experimental Results & Audits
* **Conference Paper Content:** Single-run pilot CF and CBF tables with reported MSE/RMSE (including historical rounding discrepancies).
* **Journal Novelty:**
  1. **Table 3 Exact Recalculation:** All 14 client cells recomputed to satisfy $\text{RMSE} = \sqrt{\text{MSE}}$ with zero arithmetic discrepancy.
  2. **Audit of Round 9 Anomaly:** Proved that the steep error drop at round 9 is an optimization artifact caused by Flower's `on_fit_config_fn` discrete step decay ($\eta = 0.01 \to 0.002$).
  3. **Entirely New MovieLens 100K Benchmark (Tables 5 & 6, Figure 4):** Evaluated 45 experimental configurations across 5 seeds [42, 43, 44, 45, 46], 3 federation sizes ($K \in \{2, 5, 10\}$), and 3 topologies (HFL, IP-CF, AC-IP-CF), demonstrating that HFL achieves constructive scaling (+15.2% Recall@10) while IP-CF suffers ranking degradation (-27.6% relative to HFL) and client dispersion spikes (CV surges to 0.2516).

### Sections 5 & 6: Discussion, Guidelines & Open Science
* **Conference Paper Content:** Brief conclusion suggesting future work.
* **Journal Novelty (100% NEW):**
  1. Extensive discussion on block-sparse update attenuation, non-linear coordinate drift in shared MLP heads, and the pseudo-VFL zero-padding trap.
  2. Actionable reporting guidelines for reproducible federated recommendation.
  3. Public GitHub repository link ([https://github.com/n0th1ng999/federated-recsys-alignment](https://github.com/n0th1ng999/federated-recsys-alignment)) providing data loaders, PyTorch models, and tests.

---

## 4. Checklist of Formal Editorial Requirements

- [x] **Title is distinctly new:**
  - *Conference:* "Optimizing collaborative filtering in federated learning by aligning the model with different data partitions"
  - *Journal:* "Model–Partition Alignment in Federated Recommender Systems: A Methodological Evaluation of Horizontal and Item-Partitioned Learning"
- [x] **Original conference paper is cited:** Cited in Section 1 and in the References (\citet{worldcist2026}).
- [x] **Editorial invitation statement prepared:** "Comments to the editor" includes: `Invitation to publish papers from WorldCIST'26`.
- [x] **Content expansion exceeds 30%:** Verified at **+130.6% textual expansion** with 6 major new scientific and empirical contributions.

---

## 5. Audit Recommendation

**Verdict:** **FULLY COMPLIANT WITH THE 30% EXTENSION REQUIREMENT.**  
The manuscript constitutes an authentic, substantial advancement beyond the preliminary conference report and is fully qualified for journal publication in *INFORMATICA*.
