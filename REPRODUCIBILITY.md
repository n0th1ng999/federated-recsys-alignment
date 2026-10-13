# REPRODUCIBILITY.md

## 1. System & Runtime Environment

The reproducibility benchmarks have been implemented and verified on the following environment:
* **Operating System:** macOS (Darwin 24.6.0, arm64) / Linux (Ubuntu 22.04+ compatible)
* **Python Version:** Python 3.10 to 3.13
* **PyTorch Version:** 2.6.0+ (supports CPU, Apple Silicon `mps`, and NVIDIA `cuda:0`)
* **Core Scientific Libraries:** NumPy 2.2+, Pandas 2.2+, Matplotlib 3.10+, SciPy 1.15+

---

## 2. Directory & Repository Setup

The clean, standalone reproducibility package is packaged in [`reproducibility_benchmark/`](file:///Users/tpereira/Projects/ISEP/Bolsa/EARSDemo_GASPAR/reproducibility_benchmark/) and publicly hosted on GitHub:
* **Public URL:** [https://github.com/n0th1ng999/federated-recsys-alignment](https://github.com/n0th1ng999/federated-recsys-alignment)

### Step 1: Create a Clean Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r reproducibility_benchmark/requirements.txt
```

---

## 3. Execution Commands & Benchmarks

### Option A: Complete End-to-End Execution (One-Click)
```bash
cd reproducibility_benchmark
./run_all_benchmarks.sh
```

### Option B: Individual Benchmark Modules

#### 1. Preprocess the EARS Pilot E-Commerce Dataset
```bash
python3 reproducibility_benchmark/data/ears/preprocess_ears.py
```
* **Inputs:** `client_data.csv`, `product_list.csv`, `orders.csv`, `viewed_products.csv`, `wishlist.csv`.
* **Output:** Preprocessed partitions in `data/ears/fl_datasets/`.
* **Verification:** Confirms 100 users, 25 products, 825 observed interactions ($33.0\%$ matrix density), and 1,675 unobserved pairs excluded from pointwise rating regression.

#### 2. Run EARS Pilot Micro-Federation Benchmark
```bash
python3 reproducibility_benchmark/experiments/benchmark_pilot_ears.py
```
* **Output:** Saves `reproducibility_benchmark/experiments/ears_pilot_verified_results.json`.
* **Metrics Reported:** Client-level and Macro MSE, RMSE ($\text{RMSE} = \sqrt{\text{MSE}}$ exactly), MAE, Huber loss ($\delta=1.0$), and between-client dispersion (SD, CV, Range).

#### 3. Run MovieLens 100K Multi-Seed Benchmark
```bash
python3 reproducibility_benchmark/experiments/benchmark_movielens.py
```
* **Topologies:** Behavioral Horizontal (HFL), Item-Partitioned CF (IP-CF), Active-Coordinate Renormalized (AC-IP-CF).
* **Federation Sizes:** $K \in \{2, 5, 10\}$.
* **Seeds:** 5 independent random seeds `[42, 43, 44, 45, 46]`.
* **Ranking Protocol:** Leave-one-positive-out per user; 99 negative candidates deterministically sampled from unobserved items (seed-keyed RNG: `seed * 10000 + user_id`).
* **Output:** Saves `reproducibility_benchmark/experiments/benchmark_results.json`.

#### 4. Regenerate Publication Figures (300 DPI)
```bash
python3 reproducibility_benchmark/experiments/generate_figures.py
```
* **Output Figures:**
  - `figures/conceptual-alignment.png`
  - `figures/round-trajectories.png`
  - `figures/final-mse.png`
  - `figures/movielens-benchmark.png`

#### 5. Run Verification & Test Suite
```bash
python3 tests/test_validation_suite.py
```
* **Verification:** Asserts 100% pass across ranking edge cases, leakage prevention, mathematical RMSE-MSE equivalence, and aggregation mechanics.

---

## 4. Seeds & Determinism

1. **MovieLens Train/Test Split:** Stratified 80% train / 20% test per user, seeded with `seed \in [42, 43, 44, 45, 46]`.
2. **Item and User Partitioning:** Permuted and split using `np.random.RandomState(seed)`.
3. **Model Weight Initialization:** Seeded with `torch.manual_seed(seed)`.
4. **Candidate Negatives:** Generated deterministically per user using `np.random.RandomState(seed * 10000 + int(user_id))` to ensure strictly identical candidate sets across HFL, IP-CF, and AC-IP-CF.

---

## 5. Hardware Device Acceleration

The benchmark automatically detects the optimal compute backend:
```python
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu"))
```
* **Apple Silicon:** Automatically utilizes Metal Performance Shaders (`mps`).
* **NVIDIA GPU:** Automatically utilizes CUDA (`cuda:0`).
* **Fallback:** Vectorized multi-threaded CPU execution.

---

## 6. Data & Privacy Scope

* **MovieLens 100K:** Publicly available academic benchmark dataset distributed by GroupLens under permissive research terms.
* **EARS Pilot Dataset:** Anonymized synthetic/pilot behavioral interactions (views, wishlist additions, orders) collected under the EARS project framework (ITEA-2021-21017-EARS). All user and product identifiers are integer indexes.
* **Scope Boundary:** This research models and quantifies mathematical optimization dynamics, gradient attenuation, ranking utility, and client-level dispersion. It does not evaluate empirical cryptographic defense strength or differential privacy $(\epsilon, \delta)$ mechanisms.
