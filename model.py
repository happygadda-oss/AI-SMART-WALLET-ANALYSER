"""
model.py
--------
This file contains ALL the "AI" logic for the Smart Wallet Analyzer.

It demonstrates a very simple, beginner-friendly Machine Learning workflow:

    1. Feature Engineering  -> build_features()
    2. Data Preprocessing   -> encode categorical column (Coin)
    3. Model Training       -> train_model()
    4. Prediction           -> predict_trade()

We intentionally keep this simple:
    - No deep learning / LSTMs
    - No real price prediction
    - No live market data
    - Just a small RandomForestClassifier learning from historical
      wallet trade patterns in wallet_data.csv

The model predicts ONE thing only:
    "Is a hypothetical trade likely to be profitable, based on
     patterns seen in historical trades?"

This is a LEARNING DEMO, not financial advice.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


def build_features(df: pd.DataFrame):
    """
    Build the feature table used to train the AI model.

    Features used (kept intentionally simple):
        - Coin (encoded as a number)
        - Buy_Price
        - Quantity
        - Wallet_Win_Rate   -> how often this wallet wins, historically
        - Wallet_Avg_Profit -> this wallet's average profit per trade

    Target:
        - Profitable_Trade  -> 1 if Profit > 0, else 0

    Returns:
        features_df, target_series, label_encoder, wallet_stats_df
    """
    data = df.copy()

    # --- Step 1: Make sure profit exists (it should already be calculated) ---
    if "Profit" not in data.columns:
        data["Profit"] = (data["Sell_Price"] - data["Buy_Price"]) * data["Quantity"]

    # --- Step 2: Create the ML target column ---
    data["Profitable_Trade"] = (data["Profit"] > 0).astype(int)

    # --- Step 3: Engineer wallet-level stats as features ---
    # These describe "how good is this wallet historically", which helps
    # the model use wallet behaviour as a signal, not just a single trade.
    wallet_stats = (
        data.groupby("Wallet_ID")
        .agg(
            Wallet_Win_Rate=("Profitable_Trade", "mean"),
            Wallet_Avg_Profit=("Profit", "mean"),
        )
        .reset_index()
    )

    data = data.merge(wallet_stats, on="Wallet_ID", how="left")

    # --- Step 4: Encode the categorical "Coin" column into numbers ---
    # ML models need numeric input, so we use a simple LabelEncoder.
    coin_encoder = LabelEncoder()
    data["Coin_Encoded"] = coin_encoder.fit_transform(data["Coin"])

    # --- Step 5: Select final feature columns ---
    feature_cols = [
        "Coin_Encoded",
        "Buy_Price",
        "Quantity",
        "Wallet_Win_Rate",
        "Wallet_Avg_Profit",
    ]

    X = data[feature_cols]
    y = data["Profitable_Trade"]

    return X, y, coin_encoder, wallet_stats


def train_model(df: pd.DataFrame):
    """
    Train a simple RandomForestClassifier on the historical trade data.

    Returns:
        model            -> the trained RandomForestClassifier
        accuracy         -> accuracy score on the held-out test set
        coin_encoder     -> fitted LabelEncoder for the Coin column
        wallet_stats     -> per-wallet win rate / avg profit table
    """
    X, y, coin_encoder, wallet_stats = build_features(df)

    # Split into train/test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # A small, simple Random Forest - easy to train, hard to overfit badly
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy, coin_encoder, wallet_stats


def predict_trade(model, coin_encoder, wallet_stats, coin, buy_price, quantity, wallet_id):
    """
    Predict whether a hypothetical new trade is likely to be profitable.

    Returns:
        prediction_label -> "Likely Profitable" or "Less Likely Profitable"
        confidence        -> model's confidence (%) in that prediction
    """
    # Look up this wallet's historical stats (win rate & avg profit).
    # If the wallet isn't found (edge case), fall back to dataset-wide averages.
    row = wallet_stats[wallet_stats["Wallet_ID"] == wallet_id]
    if not row.empty:
        win_rate = row["Wallet_Win_Rate"].values[0]
        avg_profit = row["Wallet_Avg_Profit"].values[0]
    else:
        win_rate = wallet_stats["Wallet_Win_Rate"].mean()
        avg_profit = wallet_stats["Wallet_Avg_Profit"].mean()

    # Encode the coin the same way it was encoded during training.
    # If it's an unseen coin, fall back to the most common encoded value.
    if coin in coin_encoder.classes_:
        coin_encoded = coin_encoder.transform([coin])[0]
    else:
        coin_encoded = 0

    input_df = pd.DataFrame(
        [{
            "Coin_Encoded": coin_encoded,
            "Buy_Price": buy_price,
            "Quantity": quantity,
            "Wallet_Win_Rate": win_rate,
            "Wallet_Avg_Profit": avg_profit,
        }]
    )

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    confidence = round(max(probabilities) * 100, 2)

    label = "Likely Profitable" if prediction == 1 else "Less Likely Profitable"

    return label, confidence
