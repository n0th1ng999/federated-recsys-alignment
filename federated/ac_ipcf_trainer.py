import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any
from models.neural_cf import NeuralCF, RatingDataset
from federated.federated_utils import evaluate_pointwise, evaluate_ranking_paired

def train_ac_ipcf(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_users: int,
    num_items: int,
    full_user_items: Dict[int, set],
    K: int,
    seed: int,
    device: torch.device,
    rounds: int = 10,
    batch_size: int = 64
) -> Dict[str, Any]:
    """
    Active-Coordinate Renormalized IP-CF (AC-IP-CF):
    - Disjoint items, shared user pool.
    - Active coordinate updates on item embeddings are renormalized over active observers:
        Delta q_i^{AC} = sum_k p_k alpha_{i,k} Delta q_{i,k} / sum_k p_k alpha_{i,k} = Delta q_{i,k}
      eliminating the 1/K attenuation on item embeddings.
    - Shared parameters (user embeddings and projection MLP) are aggregated via standard FedAvg.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    all_items = np.arange(num_items)
    rng.shuffle(all_items)
    item_splits = np.array_split(all_items, K)

    client_train_dfs = [train_df[train_df["item_id"].isin(item_splits[k])].reset_index(drop=True) for k in range(K)]
    client_test_dfs = [test_df[test_df["item_id"].isin(item_splits[k])].reset_index(drop=True) for k in range(K)]

    global_model = NeuralCF(num_users, num_items).to(device)
    criterion = nn.MSELoss()

    for round_idx in range(1, rounds + 1):
        lr = 0.01 if round_idx <= 8 else 0.002
        sample_counts = [len(df) for df in client_train_dfs]
        total_samples = sum(sample_counts)

        client_models = []
        for k in range(K):
            model = NeuralCF(num_users, num_items).to(device)
            model.load_state_dict(global_model.state_dict())

            ds = RatingDataset(
                client_train_dfs[k]["user_id"].values,
                client_train_dfs[k]["item_id"].values,
                client_train_dfs[k]["rating"].values
            )
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

        # Active-Coordinate Renormalized Aggregation
        new_global_sd = global_model.state_dict()
        shared_keys = [k for k in new_global_sd.keys() if k != "item_emb.weight"]

        # 1. Standard FedAvg for shared parameters
        for key in shared_keys:
            agg_param = sum(
                (sample_counts[k] / total_samples) * client_models[k].state_dict()[key]
                for k in range(K)
            )
            new_global_sd[key].copy_(agg_param)

        # 2. Active-coordinate renormalization for item embeddings
        new_item_emb = new_global_sd["item_emb.weight"].clone()
        for k in range(K):
            c_items = torch.tensor(item_splits[k], dtype=torch.long, device=device)
            # Active client's delta is preserved without 1/K dilution
            new_item_emb[c_items] = client_models[k].item_emb.weight.data[c_items]

        new_global_sd["item_emb.weight"].copy_(new_item_emb)
        global_model.load_state_dict(new_global_sd)

    final_client_metrics = []
    for k in range(K):
        pt = evaluate_pointwise(global_model, client_test_dfs[k], device)
        rk = evaluate_ranking_paired(
            global_model,
            client_test_dfs[k],
            full_user_items,
            num_items,
            seed=seed,
            device=device
        )
        final_client_metrics.append({**pt, **rk})

    return {
        "final_client_metrics": final_client_metrics,
        "sample_counts": sample_counts,
        "weights": [c / total_samples for c in sample_counts]
    }
