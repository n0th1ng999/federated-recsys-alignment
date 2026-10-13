import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any
from models.neural_cf import NeuralCF, RatingDataset
from federated.federated_utils import evaluate_pointwise, evaluate_ranking_paired

def train_hfl(
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
    Horizontal Federated Learning (HFL):
    - Disjoint users, shared item catalogue.
    - Dense aggregation over item embeddings and projection MLP.
    - Local user embeddings kept private to each client partition.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    all_users = np.arange(num_users)
    rng.shuffle(all_users)
    user_splits = np.array_split(all_users, K)

    client_train_dfs = [train_df[train_df["user_id"].isin(user_splits[k])].reset_index(drop=True) for k in range(K)]
    client_test_dfs = [test_df[test_df["user_id"].isin(user_splits[k])].reset_index(drop=True) for k in range(K)]

    global_model = NeuralCF(num_users, num_items).to(device)
    client_models = [NeuralCF(num_users, num_items).to(device) for _ in range(K)]
    for m in client_models:
        m.load_state_dict(global_model.state_dict())

    criterion = nn.MSELoss()
    shared_param_names = [
        "item_emb.weight",
        "fc1.weight", "fc1.bias",
        "fc2.weight", "fc2.bias",
        "output.weight", "output.bias",
    ]

    for round_idx in range(1, rounds + 1):
        lr = 0.01 if round_idx <= 8 else 0.002
        sample_counts = [len(df) for df in client_train_dfs]
        total_samples = sum(sample_counts)

        for k in range(K):
            model = client_models[k]
            for name in shared_param_names:
                model.state_dict()[name].copy_(global_model.state_dict()[name])

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

        # Dense FedAvg on shared parameters
        new_global_sd = global_model.state_dict()
        for name in shared_param_names:
            agg_param = sum(
                (sample_counts[k] / total_samples) * client_models[k].state_dict()[name]
                for k in range(K)
            )
            new_global_sd[name].copy_(agg_param)
        global_model.load_state_dict(new_global_sd)

    # Assemble full model with local user embeddings
    final_eval_model = NeuralCF(num_users, num_items).to(device)
    final_eval_model.load_state_dict(global_model.state_dict())
    full_user_emb = final_eval_model.user_emb.weight.data
    for k in range(K):
        c_users = torch.tensor(user_splits[k], dtype=torch.long, device=device)
        full_user_emb[c_users] = client_models[k].user_emb.weight.data[c_users]

    final_client_metrics = []
    for k in range(K):
        pt = evaluate_pointwise(final_eval_model, client_test_dfs[k], device)
        rk = evaluate_ranking_paired(
            final_eval_model,
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
