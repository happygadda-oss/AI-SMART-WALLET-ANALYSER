"""
user_data.py
------------
Handles per-user datasets: each registered user can upload their own
trade CSV, which is validated and saved separately from everyone else's.

Files are stored locally under the "user_data/" folder, one CSV per
username (e.g. user_data/trader1_wallet_data.csv). If a user hasn't
uploaded their own file yet, the app falls back to the shared sample
dataset (wallet_data.csv) so the dashboard still works out of the box.

This keeps things simple - no database, just per-user files on disk.
"""

import os
import pandas as pd

USER_DATA_DIR = "user_data"
DEFAULT_DATASET_PATH = "wallet_data.csv"

REQUIRED_COLUMNS = ["Wallet_ID", "Trade_Date", "Coin", "Buy_Price", "Sell_Price", "Quantity"]

os.makedirs(USER_DATA_DIR, exist_ok=True)


def get_user_csv_path(username: str) -> str:
    """Build the file path where this user's dataset is/will be stored."""
    safe_name = "".join(c for c in username if c.isalnum() or c in ("_", "-"))
    return os.path.join(USER_DATA_DIR, f"{safe_name}_wallet_data.csv")


def has_user_dataset(username: str) -> bool:
    """Check whether this user has already uploaded their own dataset."""
    return os.path.exists(get_user_csv_path(username))


def validate_csv(uploaded_file) -> tuple:
    """
    Validate an uploaded CSV has the required columns and readable data.
    Returns (is_valid: bool, message: str, dataframe_or_None).
    """
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        return False, f"Could not read this file as a CSV: {e}", None

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        return False, f"Missing required column(s): {', '.join(missing_cols)}", None

    if len(df) == 0:
        return False, "The uploaded file has no trade rows.", None

    return True, "File looks good!", df


def save_user_dataset(username: str, df: pd.DataFrame) -> None:
    """Save a validated dataframe as this user's dataset on disk."""
    df.to_csv(get_user_csv_path(username), index=False)


def delete_user_dataset(username: str) -> None:
    """Remove this user's uploaded dataset (falls back to sample data)."""
    path = get_user_csv_path(username)
    if os.path.exists(path):
        os.remove(path)


def get_active_dataset_path(username: str) -> str:
    """
    Return the CSV path that should be loaded for this user:
    their own uploaded dataset if it exists, otherwise the shared
    sample dataset.
    """
    user_path = get_user_csv_path(username)
    if os.path.exists(user_path):
        return user_path
    return DEFAULT_DATASET_PATH
