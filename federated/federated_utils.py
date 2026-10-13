import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any

def evaluate_pointwise(model: nn.Module, test_df, device: torch.device) -> Dict[str, float]:
    """Compute exact MSE, RMSE, MAE, and Huber loss (delta=1.0)."""
    if len(test_df) == 0:
        return {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "huber": 0.0}

    model.eval()
    u_tensor = torch.tensor(test_df["user_id"].values, dtype=torch.long, device=device)
    i_tensor = torch.tensor(test_df["item_id"].values, dtype=torch.long, device=device)
    y_true = test_df["rating"].values.astype(np.float64)

    with torch.no_grad():
        preds = model(u_tensor, i_tensor).cpu().numpy().astype(np.float64)

    residuals = y_true - preds
    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))

    # Huber loss (delta = 1.0)
    abs_err = np.abs(residuals)
    huber_mask = abs_err <= 1.0
    huber_err = np.where(huber_mask, 0.5 * (residuals ** 2), abs_err - 0.5)
    huber = float(np.mean(huber_err))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "huber": huber,
    }


def evaluate_ranking_paired(
    model: nn.Module,
    client_test_df,
    full_user_items: Dict[int, set],
    num_items: int,
    seed: int,
    device: torch.device,
    k_eval: int = 10
) -> Dict[str, float]:
    """
    Deterministic paired ranking protocol:
    For each test user, evaluates the user's highest rated test item against 99 unobserved negatives.
    Negatives are sampled strictly from items not interacted with in train or test.
    RNG is keyed to (seed * 10000 + user_id) ensuring identical paired evaluation across topologies.
    """
    if len(client_test_df) == 0:
        return {f"recall@{k_eval}": 0.0, f"ndcg@{k_eval}": 0.0}

    unique_users = client_test_df["user_id"].unique()
    all_items = np.arange(num_items)

    model.eval()
    recalls = []
    ndcgs = []

    user_batch_size = 128
    for start_idx in range(0, len(unique_users), user_batch_size):
        batch_users = unique_users[start_idx : start_idx + user_batch_size]

        eval_users = []
        eval_items = []

        for u in batch_users:
            u_test = client_test_df[client_test_df["user_id"] == u]
            best_row = u_test.sort_values(by=["rating", "timestamp" if "timestamp" in u_test.columns else "rating"], ascending=[False, False]).iloc[0]
            pos_item = int(best_row["item_id"])

            interacted = full_user_items.get(u, set())
            unobserved = np.setdiff1d(all_items, list(interacted))

            user_rng = np.random.RandomState(seed * 10000 + int(u))
            if len(unobserved) < 99:
                sampled_negs = user_rng.choice(unobserved, size=len(unobserved), replace=False)
            else:
                sampled_negs = user_rng.choice(unobserved, size=99, replace=False)

            candidate_items = [pos_item] + list(sampled_negs)
            eval_users.extend([u] * len(candidate_items))
            eval_items.extend(candidate_items)

        eval_users_t = torch.tensor(eval_users, dtype=torch.long, device=device)
        eval_items_t = torch.tensor(eval_items, dtype=torch.long, device=device)

        with torch.no_grad():
            scores = model(eval_users_t, eval_items_t).cpu().numpy()

        offset = 0
        for u in batch_users:
            interacted = full_user_items.get(u, set())
            n_candidates = min(100, 1 + (num_items - len(interacted)))
            u_scores = scores[offset : offset + n_candidates]
            offset += n_candidates

            pos_score = u_scores[0]
            greater = np.sum(u_scores[1:] > pos_score)
            rank = 1 + greater

            if rank <= k_eval:
                recalls.append(1.0)
                ndcgs.append(1.0 / np.log2(rank + 1))
            else:
                recalls.append(0.0)
                ndcgs.append(0.0)

    return {
        f"recall@{k_eval}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{k_eval}": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }


def compute_dispersion(values: List[float]) -> Dict[str, float]:
    """Compute Macro mean, sample standard deviation (ddof=1), CV, and Range."""
    arr = np.array(values, dtype=np.float64)
    mean_val = float(np.mean(arr))
    if len(arr) > 1:
        sd_val = float(np.std(arr, ddof=1))
    else:
        sd_val = 0.0
    cv_val = float(sd_val / mean_val) if abs(mean_val) > 1e-12 else 0.0
    range_val = float(np.max(arr) - np.min(arr)) if len(arr) > 0 else 0.0
    return {
        "mean": mean_val,
        "sd": sd_val,
        "cv": cv_val,
        "range": range_val,
    }
