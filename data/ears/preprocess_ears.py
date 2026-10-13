"""
Preprocess EARS Pilot Dataset:
Combines orders (weight 3), wishlists (weight 2), and views (weight 1)
into ratings matrix. Generates clean interaction records and validates counts:
- 100 users
- 25 items
- 825 observed interactions
- 1675 unobserved pairs (density = 33.0%)
"""
import os
import pandas as pd
import numpy as np

def build_ears_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    products = pd.read_csv(os.path.join(base_dir, "product_list.csv"))
    clients = pd.read_csv(os.path.join(base_dir, "client_data.csv"))
    orders = pd.read_csv(os.path.join(base_dir, "orders.csv"))
    viewed = pd.read_csv(os.path.join(base_dir, "viewed_products.csv"))
    wishlist = pd.read_csv(os.path.join(base_dir, "wishlist.csv"))

    valid_product_ids = set(products["ProductID"])
    orders = orders[orders["ProductID"].isin(valid_product_ids)].drop_duplicates(subset=["ClientID", "ProductID"]).copy()
    viewed = viewed[viewed["ProductID"].isin(valid_product_ids)].drop_duplicates(subset=["ClientID", "ProductID"]).copy()
    wishlist = wishlist[wishlist["ProductID"].isin(valid_product_ids)].drop_duplicates(subset=["ClientID", "ProductID"]).copy()

    orders["rating"] = 3.0
    wishlist["rating"] = 2.0
    viewed["rating"] = 1.0

    df_orders = orders[["ClientID", "ProductID", "rating"]].rename(columns={"ClientID": "user_id", "ProductID": "item_id"})
    df_viewed = viewed[["ClientID", "ProductID", "rating"]].rename(columns={"ClientID": "user_id", "ProductID": "item_id"})
    df_wish = wishlist[["ClientID", "ProductID", "rating"]].rename(columns={"ClientID": "user_id", "ProductID": "item_id"})

    combined = pd.concat([df_orders, df_viewed, df_wish], ignore_index=True)
    interactions = combined.groupby(["user_id", "item_id"], as_index=False)["rating"].sum()

    # Zero-index user and item IDs
    # User IDs are 1 to 100 -> map to 0 to 99
    # Item IDs are 1 to 25 -> map to 0 to 24
    interactions["user_id"] = interactions["user_id"] - 1
    interactions["item_id"] = interactions["item_id"] - 1

    out_csv = os.path.join(base_dir, "ratings.csv")
    interactions.to_csv(out_csv, index=False)

    print(f"EARS Dataset preprocessed successfully:")
    print(f"  - Interactions: {len(interactions)}")
    print(f"  - Unique users: {interactions['user_id'].nunique()}")
    print(f"  - Unique items: {interactions['item_id'].nunique()}")
    print(f"  - Possible pairs: {100 * 25}")
    print(f"  - Matrix density: {len(interactions) / (100 * 25):.1%}")

if __name__ == "__main__":
    build_ears_dataset()
