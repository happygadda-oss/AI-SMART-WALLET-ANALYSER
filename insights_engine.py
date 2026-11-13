"""
insights_engine.py
-------------------
Automated, cross-wallet insight generation.

This scans the ENTIRE dataset (not just one wallet at a time, like the
"Wallet Insights" section in app.py does) and surfaces plain-English
observations such as:

    "Wallet W001 earns 70% of its profits from BTC."
    "Wallet W003 performs poorly in SOL."
    "Wallet W005 has the highest consistency."
    "Wallet W002 trades less but earns more per trade."

⚠️ Important: this is RULE-BASED pattern detection (statistics + simple
thresholds), not a machine learning model. It's a different kind of "AI"
than model.py's RandomForestClassifier - more like an automated analyst
that applies a checklist of rules to the data. We keep it in its own file
so it's clear which parts of the project are real ML vs. rule-based logic.

Each rule below looks for one specific pattern:
    1. Coin concentration - is most of a wallet's profit coming from
       one single coin?
    2. Weak coin          - which coin is dragging a wallet's
       performance down?
    3. Highest consistency - which wallet has the single best win rate?
    4. Efficiency          - which wallets trade less often but earn
       more, on average, per trade?

Tune the thresholds (marked below) to make insights more or less strict.
"""

import pandas as pd

# --- Thresholds you can tune ---
CONCENTRATION_THRESHOLD = 50   # % of profit from one coin to count as "concentrated"
MIN_TRADES_FOR_COIN_INSIGHT = 3  # ignore coins a wallet has traded fewer than N times


def _wallet_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build one summary row per wallet: trades, profit, avg profit, win rate."""
    return (
        df.groupby("Wallet_ID")
        .agg(
            Total_Trades=("Profit", "count"),
            Total_Profit=("Profit", "sum"),
            Avg_Profit=("Profit", "mean"),
            Win_Rate=("Profit", lambda x: (x > 0).mean() * 100),
        )
        .reset_index()
    )


def _coin_concentration_insights(df: pd.DataFrame) -> list:
    """Find wallets where most profit comes from a single coin."""
    insights = []
    for wallet in sorted(df["Wallet_ID"].unique()):
        w_df = df[df["Wallet_ID"] == wallet]
        coin_profit = w_df.groupby("Coin")["Profit"].sum()
        positive = coin_profit[coin_profit > 0]

        total_positive = positive.sum()
        if total_positive <= 0:
            continue  # this wallet has no profitable coins at all

        top_coin = positive.idxmax()
        share = positive[top_coin] / total_positive * 100

        if share >= CONCENTRATION_THRESHOLD:
            insights.append({
                "wallet": wallet,
                "type": "Coin Concentration",
                "icon": "🎯",
                "text": f"Wallet {wallet} earns {share:.0f}% of its profits from {top_coin}.",
            })
    return insights


def _weak_coin_insights(df: pd.DataFrame) -> list:
    """Find each wallet's worst-performing coin, if it's a net loss."""
    insights = []
    for wallet in sorted(df["Wallet_ID"].unique()):
        w_df = df[df["Wallet_ID"] == wallet]
        coin_stats = w_df.groupby("Coin").agg(
            Profit=("Profit", "sum"),
            Win_Rate=("Profit", lambda x: (x > 0).mean() * 100),
            Trades=("Profit", "count"),
        )
        # ignore coins with too few trades - not enough evidence yet
        coin_stats = coin_stats[coin_stats["Trades"] >= MIN_TRADES_FOR_COIN_INSIGHT]
        if coin_stats.empty:
            continue

        worst_coin = coin_stats["Profit"].idxmin()
        row = coin_stats.loc[worst_coin]

        if row["Profit"] < 0:
            insights.append({
                "wallet": wallet,
                "type": "Weak Coin",
                "icon": "⚠️",
                "text": (
                    f"Wallet {wallet} performs poorly in {worst_coin} "
                    f"(win rate {row['Win_Rate']:.0f}%, total profit ${row['Profit']:,.2f})."
                ),
            })
    return insights


def _consistency_insight(summary: pd.DataFrame) -> list:
    """Identify the single wallet with the highest win rate across all wallets."""
    if summary.empty:
        return []
    best = summary.loc[summary["Win_Rate"].idxmax()]
    return [{
        "wallet": best["Wallet_ID"],
        "type": "Consistency",
        "icon": "🏅",
        "text": f"Wallet {best['Wallet_ID']} has the highest consistency (win rate {best['Win_Rate']:.0f}%).",
    }]


def _efficiency_insights(summary: pd.DataFrame) -> list:
    """Find wallets that trade less often than average but earn more per trade than average."""
    if summary.empty:
        return []

    avg_trades = summary["Total_Trades"].mean()
    avg_profit_per_trade = summary["Avg_Profit"].mean()

    efficient = summary[
        (summary["Total_Trades"] < avg_trades) & (summary["Avg_Profit"] > avg_profit_per_trade)
    ]

    insights = []
    for _, row in efficient.iterrows():
        insights.append({
            "wallet": row["Wallet_ID"],
            "type": "Efficiency",
            "icon": "⚡",
            "text": (
                f"Wallet {row['Wallet_ID']} trades less ({int(row['Total_Trades'])} trades) "
                f"but earns more per trade (avg ${row['Avg_Profit']:,.2f})."
            ),
        })
    return insights


def generate_insights(df: pd.DataFrame) -> list:
    """
    Run all rules across the full dataset and return a combined list of
    insight dicts: {"wallet": ..., "type": ..., "icon": ..., "text": ...}
    """
    summary = _wallet_summary(df)

    insights = []
    insights += _coin_concentration_insights(df)
    insights += _weak_coin_insights(df)
    insights += _consistency_insight(summary)
    insights += _efficiency_insights(summary)

    return insights
