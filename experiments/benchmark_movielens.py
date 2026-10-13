"""
MovieLens 100K Federated Recommendation Benchmark:
Evaluates Behavioral Horizontal (HFL), Item-Partitioned (IP-CF), and Active-Coordinate Renormalized (AC-IP-CF)
across federation sizes K in {2, 5, 10} over 5 independent random seeds [42, 43, 44, 45, 46].
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Any

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from federated.hfl_trainer import train_hfl
from federated.ipcf_trainer import train_ipcf
from federated.ac_ipcf_trainer import train_ac_ipcf
from federated.federated_utils import compute_dispersion

def load_movielens_100k(data_path: str):
    df = pd.read_csv(data_path, sep="\t", names=["user_id", "item_id", "rating", "timestamp"], engine="c")
    df["user_id"] = df["user_id"] - 1
    df["item_id"] = df["item_id"] - 1
    df["rating"] = df["rating"].astype(np.float32)
    return df

def split_train_test(df: pd.DataFrame, seed: int):
    rng = np.random.RandomState(seed)
    train_indices = []
    test_indices = []
    for _, group in df.groupby("user_id"):
        idx = group.index.values.copy()
        rng.shuffle(idx)
        n_train = int(len(idx) * 0.8)
        train_indices.extend(idx[:n_train])
        test_indices.extend(idx[n_train:])
    return df.loc[train_indices].reset_index(drop=True), df.loc[test_indices].reset_index(drop=True)

def run_benchmark():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"Executing MovieLens 100K benchmark on device: {device}")

    data_path = os.path.join(project_root, "data", "ml-100k", "u.data")
    df = load_movielens_100k(data_path)
    num_users = int(df["user_id"].max() + 1)
    num_items = int(df["item_id"].max() + 1)

    full_user_items = df.groupby("user_id")["item_id"].apply(set).to_dict()

    seeds = [42, 43, 44, 45, 46]
    k_values = [2, 5, 10]
    topologies = ["HFL", "IP-CF", "AC-IP-CF"]

    results = {}

    for k in k_values:
        for topo in topologies:
            key = f"{topo}_K{k}"
            print(f"\n--- Running {key} across {len(seeds)} seeds ---")
            seed_client_metrics = []

            for s in seeds:
                train_df, test_df = split_train_test(df, seed=s)
                if topo == "HFL":
                    res = train_hfl(train_df, test_df, num_users, num_items, full_user_items, K=k, seed=s, device=device)
                elif topo == "IP-CF":
                    res = train_ipcf(train_df, test_df, num_users, num_items, full_user_items, K=k, seed=s, device=device)
                elif topo == "AC-IP-CF":
                    res = train_ac_ipcf(train_df, test_df, num_users, num_items, full_user_items, K=k, seed=s, device=device)

                seed_client_metrics.append(res["final_client_metrics"])

            # Compute macro metrics per seed, then mean +- SD across seeds
            seed_macro_mse = []
            seed_macro_rmse = []
            seed_macro_mae = []
            seed_macro_recall = []
            seed_macro_ndcg = []

            seed_mse_sd = []
            seed_mse_cv = []
            seed_mse_range = []

            seed_recall_sd = []
            seed_recall_cv = []

            for c_metrics in seed_client_metrics:
                mses = [m["mse"] for m in c_metrics]
                rmses = [m["rmse"] for m in c_metrics]
                maes = [m["mae"] for m in c_metrics]
                recalls = [m["recall@10"] for m in c_metrics]
                ndcgs = [m["ndcg@10"] for m in c_metrics]

                seed_macro_mse.append(np.mean(mses))
                seed_macro_rmse.append(np.mean(rmses))
                seed_macro_mae.append(np.mean(maes))
                seed_macro_recall.append(np.mean(recalls))
                seed_macro_ndcg.append(np.mean(ndcgs))

                disp_mse = compute_dispersion(mses)
                seed_mse_sd.append(disp_mse["sd"])
                seed_mse_cv.append(disp_mse["cv"])
                seed_mse_range.append(disp_mse["range"])

                disp_rec = compute_dispersion(recalls)
                seed_recall_sd.append(disp_rec["sd"])
                seed_recall_cv.append(disp_rec["cv"])

            summary = {
                "macro_mse_mean": float(np.mean(seed_macro_mse)),
                "macro_mse_sd": float(np.std(seed_macro_mse, ddof=1)),
                "macro_rmse_mean": float(np.mean(seed_macro_rmse)),
                "macro_rmse_sd": float(np.std(seed_macro_rmse, ddof=1)),
                "macro_mae_mean": float(np.mean(seed_macro_mae)),
                "macro_mae_sd": float(np.std(seed_macro_mae, ddof=1)),
                "recall10_mean": float(np.mean(seed_macro_recall)),
                "recall10_sd": float(np.std(seed_macro_recall, ddof=1)),
                "ndcg10_mean": float(np.mean(seed_macro_ndcg)),
                "ndcg10_sd": float(np.std(seed_macro_ndcg, ddof=1)),

                "client_mse_sd_mean": float(np.mean(seed_mse_sd)),
                "client_mse_sd_sd": float(np.std(seed_mse_sd, ddof=1)),
                "client_mse_cv_mean": float(np.mean(seed_mse_cv)),
                "client_mse_cv_sd": float(np.std(seed_mse_cv, ddof=1)),
                "client_mse_range_mean": float(np.mean(seed_mse_range)),
                "client_mse_range_sd": float(np.std(seed_mse_range, ddof=1)),

                "client_recall_sd_mean": float(np.mean(seed_recall_sd)),
                "client_recall_sd_sd": float(np.std(seed_recall_sd, ddof=1)),
                "client_recall_cv_mean": float(np.mean(seed_recall_cv)),
                "client_recall_cv_sd": float(np.std(seed_recall_cv, ddof=1)),
            }
            results[key] = summary
            print(f"  MSE: {summary['macro_mse_mean']:.4f} +- {summary['macro_mse_sd']:.4f} | Recall@10: {summary['recall10_mean']:.4f} +- {summary['recall10_sd']:.4f} | Recall CV: {summary['client_recall_cv_mean']:.4f}")

    out_file = os.path.join(project_root, "experiments", "benchmark_results.json")
    with open(out_file, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nAll benchmark results saved to {out_file}")

if __name__ == "__main__":
    run_benchmark()
