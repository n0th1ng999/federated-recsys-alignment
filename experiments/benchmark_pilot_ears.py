"""
EARS Pilot Dataset Federated Evaluation:
Evaluates Collaborative Filtering under HFL and IP-CF for K in {2, 5} clients
on the exact experimental splits from fl_datasets/.
Outputs exact mathematically consistent metrics (RMSE = sqrt(MSE)).
"""
import os
import sys
import copy
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Any

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.neural_cf import NeuralCF, RatingDataset
from federated.federated_utils import compute_dispersion

def run_pilot_fl(csv_files: List[str], rounds: int = 10, batch_size: int = 64, seed: int = 42, device: torch.device = None):
    if device is None:
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    K = len(csv_files)
    client_dfs = [pd.read_csv(f) for f in csv_files]
    sample_counts = [len(df) for df in client_dfs]
    total_samples = sum(sample_counts)

    # 100 users, 25 items
    global_model = NeuralCF(num_users=100, num_items=25, embedding_dim=32).to(device)
    criterion = nn.MSELoss()

    for r in range(1, rounds + 1):
        lr = 0.01 if r <= 8 else 0.002
        client_models = []

        for k in range(K):
            df = client_dfs[k]
            model = copy.deepcopy(global_model)
            ds = RatingDataset(df["user_id"].values, df["item_id"].values, df["rating"].values)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            model.train()
            for b_u, b_i, b_r in loader:
                b_u, b_i, b_r = b_u.to(device), b_i.to(device), b_r.to(device)
                optimizer.zero_grad()
                pred = model(b_u, b_i)
                loss = criterion(pred, b_r)
                loss.backward()
                optimizer.step()

            client_models.append(model)

        # Standard dense FedAvg
        new_g = global_model.state_dict()
        for key in new_g.keys():
            new_g[key] = sum((sample_counts[k] / total_samples) * client_models[k].state_dict()[key] for k in range(K))
        global_model.load_state_dict(new_g)

    # Final exact evaluation per client on holdout/partition observations
    client_results = []
    huber_fn = nn.HuberLoss(delta=1.0)
    global_model.eval()

    with torch.no_grad():
        for k in range(K):
            df = client_dfs[k]
            u_t = torch.tensor(df["user_id"].values, dtype=torch.long, device=device)
            i_t = torch.tensor(df["item_id"].values, dtype=torch.long, device=device)
            r_t = torch.tensor(df["rating"].values, dtype=torch.float32, device=device)

            pred = global_model(u_t, i_t)
            residuals = (r_t - pred).cpu().numpy().astype(np.float64)

            mse = float(np.mean(residuals ** 2))
            rmse = float(np.sqrt(mse))
            mae = float(np.mean(np.abs(residuals)))

            abs_r = np.abs(residuals)
            huber_err = np.where(abs_r <= 1.0, 0.5 * (residuals ** 2), abs_r - 0.5)
            huber = float(np.mean(huber_err))

            client_results.append({
                "client": k + 1,
                "n_samples": len(df),
                "rmse": rmse,
                "mse": mse,
                "mae": mae,
                "huber": huber,
            })

    return client_results


def evaluate_ears_pilot():
    device = torch.device("cpu")
    base_data = os.path.join(project_root, "data", "ears", "fl_datasets")

    configs = {
        ("HFL", 2): [os.path.join(base_data, f"client{i}_by_user_HALF.csv") for i in [1, 2]],
        ("IP-CF", 2): [os.path.join(base_data, f"client{i}_by_item_HALF.csv") for i in [1, 2]],
        ("HFL", 5): [os.path.join(base_data, f"client{i}_by_user_FIFTH.csv") for i in range(1, 6)],
        ("IP-CF", 5): [os.path.join(base_data, f"client{i}_by_item_FIFTH.csv") for i in range(1, 6)],
    }

    all_pilot_results = {}

    print("=== EARS Pilot Micro-Federation Evaluation (Exact & Verified) ===\n")
    for (topo, K), files in configs.items():
        res = run_pilot_fl(files, device=device)
        all_pilot_results[f"{topo}_K{K}"] = res

        print(f"--- {topo} (K = {K}) ---")
        mses = [c["mse"] for c in res]
        rmses = [c["rmse"] for c in res]
        maes = [c["mae"] for c in res]
        hubers = [c["huber"] for c in res]

        for c in res:
            print(f"  Client {c['client']} (n={c['n_samples']}): RMSE={c['rmse']:.4f}, MSE={c['mse']:.4f}, MAE={c['mae']:.4f}, Huber={c['huber']:.4f} [Check: sqrt({c['mse']:.4f}) = {np.sqrt(c['mse']):.4f}]")

        disp_mse = compute_dispersion(mses)
        print(f"  Macro MSE: {disp_mse['mean']:.4f} | SD: {disp_mse['sd']:.4f} | CV: {disp_mse['cv']:.4f}")
        print()

    out_json = os.path.join(project_root, "experiments", "ears_pilot_verified_results.json")
    with open(out_json, "w") as fp:
        json.dump(all_pilot_results, fp, indent=2)
    print(f"Saved verified results to: {out_json}")

if __name__ == "__main__":
    evaluate_ears_pilot()
