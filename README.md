# Model–Partition Alignment in Federated Recommender Systems: Reproducibility Package

This repository contains the complete experimental code, dataset loaders, models, and evaluation scripts supporting the journal article:

> **Model–Partition Alignment in Federated Recommender Systems: A Methodological Evaluation of Horizontal and Item-Partitioned Learning**  
> *Tiago Pereira, João Gaspar, Diogo Gonçalves, José Pessoa, Bruno Ribeiro, Diogo Martinho, Joaquim Santos, Goreti Marreiros*  
> *INFORMATICA (Vilnius University), 2026.*

---

## 1. Overview of Topologies & Baselines

This benchmark evaluates federated neural collaborative filtering across three core topological configurations:
1. **Behavioral Horizontal Federated Learning (HFL):** Disjoint user partitions observing a shared item catalogue, aggregated via standard sample-weighted dense FedAvg.
2. **Item-Partitioned Collaborative Filtering (IP-CF):** Disjoint item partitions with a shared user pool, aggregated via standard dense FedAvg (demonstrating the theoretical $p_k \approx 1/K$ coordinate attenuation formalized in Proposition 1).
3. **Active-Coordinate Renormalized IP-CF (AC-IP-CF):** Disjoint item partitions where active item coordinate rows are renormalized strictly over active client updates ($\Delta \boldsymbol{q}_i^{\text{AC}} = \Delta \boldsymbol{q}_{i,k}$), isolating the impact of multi-layer representation drift in shared projection layers.

---

## 2. Directory Structure

```text
reproducibility_benchmark/
├── README.md                      # Complete documentation
├── requirements.txt               # Python package dependencies
├── run_all_benchmarks.sh          # One-click execution script
├── data/
│   ├── ears/                      # EARS E-commerce pilot dataset
│   │   ├── client_data.csv        # 100 client demographic records
│   │   ├── product_list.csv       # 25 product catalogue entries
│   │   ├── orders.csv             # 300 purchase records (weight 3)
│   │   ├── viewed_products.csv    # 500 product view events (weight 1)
│   │   ├── wishlist.csv           # 200 wishlist additions (weight 2)
│   │   ├── fl_datasets/           # Partition splits for 2 and 5 clients
│   │   └── preprocess_ears.py     # Aggregates events into 825 unique interactions
│   └── ml-100k/
│       └── u.data                 # MovieLens 100K ratings (100,000 ratings, 943 users, 1,682 movies)
├── models/
│   ├── __init__.py
│   └── neural_cf.py               # PyTorch Neural CF (Embedding 32 + 2-layer MLP)
├── federated/
│   ├── __init__.py
│   ├── hfl_trainer.py             # Horizontal Federated Learning
│   ├── ipcf_trainer.py            # Standard Item-Partitioned CF (Dense FedAvg)
│   ├── ac_ipcf_trainer.py         # Active-Coordinate Renormalized IP-CF (AC-FedAvg)
│   └── federated_utils.py         # Pointwise errors, 1:99 paired ranking, and dispersion
└── experiments/
    ├── benchmark_movielens.py     # MovieLens 100K benchmark across K in {2, 5, 10} and 5 seeds
    ├── benchmark_pilot_ears.py    # EARS Pilot micro-federation evaluation
    └── generate_figures.py        # Publication-ready figure generation (300 DPI)
```

---

## 3. Installation & Setup

We recommend creating a fresh Python virtual environment (Python 3.10+):

```bash
# Create and activate environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 4. Running the Benchmarks

### Option A: Run Everything (One Command)
```bash
./run_all_benchmarks.sh
```

### Option B: Run Specific Benchmarks Individually

1. **Preprocess EARS Pilot Dataset:**
   ```bash
   python3 data/ears/preprocess_ears.py
   ```
   *Validates:* 100 users, 25 items, 825 observed interactions, 1,675 unobserved pairs (density = 33.0%).

2. **Evaluate EARS Pilot Micro-Federation:**
   ```bash
   python3 experiments/benchmark_pilot_ears.py
   ```
   *Outputs:* Exact per-client and macro MSE, RMSE ($\text{RMSE} = \sqrt{\text{MSE}}$), MAE, and Huber loss, alongside dispersion standard deviation and coefficient of variation.

3. **Run MovieLens 100K Expanded Multi-Seed Benchmark:**
   ```bash
   python3 experiments/benchmark_movielens.py
   ```
   *Outputs:* Multi-seed results across $K \in \{2, 5, 10\}$ for HFL, IP-CF, and AC-IP-CF over 5 random seeds [42, 43, 44, 45, 46]. Saves `benchmark_results.json`.

4. **Regenerate Publication Figures (300 DPI):**
   ```bash
   python3 experiments/generate_figures.py
   ```
   *Generates:*
   - `figures/conceptual-alignment.png` (4-panel taxonomy)
   - `figures/round-trajectories.png` (learning rate step-decay audit)
   - `figures/final-mse.png` (EARS pilot separated evaluation units)
   - `figures/movielens-benchmark.png` (scaling and dispersion panel across $K \in \{2, 5, 10\}$)

---

## 5. Ranking Evaluation Protocol

To ensure rigorous evaluation without data leakage:
- **Test Target:** For each user with test ratings, the target test item is chosen as the user's highest rated test item.
- **Negative Candidates:** 99 candidate items are sampled uniformly from the complete catalogue strictly excluding all items the user interacted with in either train or test sets.
- **Deterministic Pairing:** Candidate negative sets are generated deterministically per random seed and user ID (`seed * 10000 + user_id`), ensuring that HFL, IP-CF, and AC-IP-CF evaluate the identical 100 candidate items for every test instance.
- **Aggregation:** Metrics are calculated per interaction, averaged within each user, and macro-averaged across clients.

---

## 6. License & Citation

Released under the Apache 2.0 License.  
**Authors:** Tiago Pereira, João Gaspar, Diogo Gonçalves, José Pessoa, Bruno Ribeiro, Diogo Martinho, Joaquim Santos, Goreti Marreiros (*Instituto Superior de Engenharia do Porto, Polytechnic of Porto, Portugal*).

If you use this benchmark suite, please cite:
```bibtex
@article{pereira2026alignment,
  title={Model--Partition Alignment in Federated Recommender Systems: A Methodological Evaluation of Horizontal and Item-Partitioned Learning},
  author={Pereira, Tiago and Gaspar, Jo{\~a}o and Gon{\c{c}}alves, Diogo and Pessoa, Jos{\'e} and Ribeiro, Bruno and Martinho, Diogo and Santos, Joaquim and Marreiros, Goreti},
  journal={INFORMATICA},
  year={2026}
}
```
