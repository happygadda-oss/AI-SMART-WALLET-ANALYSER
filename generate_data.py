"""
generate_data.py
-----------------
One-off helper script used to create the sample 'wallet_data.csv' dataset.
This is NOT part of the final app - it's only used once to generate
realistic-looking sample data for the project.

Run: python generate_data.py
"""

import random
import csv
from datetime import datetime, timedelta

random.seed(42)  # fixed seed so the dataset is reproducible

# 8 sample wallet IDs
WALLETS = [f"W{str(i).zfill(3)}" for i in range(1, 9)]

# A small set of realistic coins with rough price ranges (min_price, max_price)
COINS = {
    "BTC": (25000, 45000),
    "ETH": (1500, 3200),
    "SOL": (20, 120),
    "BNB": (200, 450),
    "ADA": (0.25, 0.75),
    "XRP": (0.4, 0.9),
    "DOGE": (0.06, 0.15),
    "MATIC": (0.5, 1.4),
}

# Give each wallet a "skill level" that biases how likely its trades are to be profitable.
# This makes some wallets consistently good, some average, some risky - which is
# useful later for the AI model and the wallet scoring/recommendation logic.
WALLET_SKILL = {
    "W001": 0.80,  # very skilled / consistent
    "W002": 0.72,
    "W003": 0.60,
    "W004": 0.55,
    "W005": 0.45,
    "W006": 0.35,
    "W007": 0.30,  # risky wallet
    "W008": 0.50,
}

NUM_TRADES = 150
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 12, 31)


def random_date():
    """Return a random date between START_DATE and END_DATE."""
    delta_days = (END_DATE - START_DATE).days
    return START_DATE + timedelta(days=random.randint(0, delta_days))


def generate_trade(wallet_id):
    """Generate a single realistic trade row for a given wallet."""
    coin = random.choice(list(COINS.keys()))
    low, high = COINS[coin]

    buy_price = round(random.uniform(low, high), 4)

    # Wallet skill influences whether the trade tends to be a win or a loss.
    skill = WALLET_SKILL[wallet_id]
    is_win = random.random() < skill

    if is_win:
        # Sell price is somewhat higher than buy price (a winning trade)
        pct_change = random.uniform(0.02, 0.35)
        sell_price = round(buy_price * (1 + pct_change), 4)
    else:
        # Sell price is somewhat lower than buy price (a losing trade)
        pct_change = random.uniform(0.02, 0.25)
        sell_price = round(buy_price * (1 - pct_change), 4)

    # Quantity depends roughly on coin price so trade sizes look realistic
    if buy_price > 1000:
        quantity = round(random.uniform(0.01, 0.5), 4)
    elif buy_price > 10:
        quantity = round(random.uniform(1, 50), 3)
    else:
        quantity = round(random.uniform(100, 5000), 2)

    trade_date = random_date().strftime("%Y-%m-%d")

    return {
        "Wallet_ID": wallet_id,
        "Trade_Date": trade_date,
        "Coin": coin,
        "Buy_Price": buy_price,
        "Sell_Price": sell_price,
        "Quantity": quantity,
    }


def main():
    rows = []
    trades_per_wallet = NUM_TRADES // len(WALLETS)
    remainder = NUM_TRADES % len(WALLETS)

    for i, wallet in enumerate(WALLETS):
        count = trades_per_wallet + (1 if i < remainder else 0)
        for _ in range(count):
            rows.append(generate_trade(wallet))

    # sort by date so the CSV looks natural
    rows.sort(key=lambda r: r["Trade_Date"])

    with open("wallet_data.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Wallet_ID", "Trade_Date", "Coin", "Buy_Price", "Sell_Price", "Quantity"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} trades across {len(WALLETS)} wallets -> wallet_data.csv")


if __name__ == "__main__":
    main()
