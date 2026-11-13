# 💹 Smart Wallet Analyzer with AI Trade Recommendation

A beginner-friendly **Data Science + Machine Learning** project that simulates the
analysis of cryptocurrency wallets using a small local dataset, and demonstrates a
basic AI-assisted trade recommendation workflow.

> ⚠️ **This is a learning / demo project.** It does **not** connect to any real
> blockchain, exchange, or crypto API. All data is simulated. Nothing here is
> financial advice.

---

## 1. Project Overview

The app lets you:

- Load a sample dataset of ~150 crypto trades across 8 wallets
- Automatically calculate profit, profit %, and win/loss outcome for every trade
- View per-wallet dashboards with key metrics and charts
- Compare all wallets side-by-side using a simple 0–100 "Wallet Score"
- Get human-readable insights (best/worst coin, consistency, etc.)
- Use a small **RandomForestClassifier** to predict whether a hypothetical new
  trade is *likely* or *less likely* to be profitable, based on historical
  patterns

The goal is to demonstrate a full, simple, end-to-end Data Science workflow —
data loading, feature engineering, visualization, scoring logic, and a basic
ML model — in a project that's small enough to read and understand in a few
hours.

---

## 2. Folder Structure

```
wallet-analyzer/
│── app.py             # Main Streamlit dashboard (UI + charts + logic)
│── auth.py             # Self-registration + login system (max 3 users)
│── styles.py            # Helper that loads external .css files into Streamlit
│── style.css            # Main dashboard theme (cards, buttons, sidebar, tables)
│── login_style.css      # Styling for the login / register screen
│── user_data.py         # Per-user dataset upload, validation & storage
│── model.py            # AI/ML logic: feature engineering, training, prediction
│── insights_engine.py   # Rule-based automated insight generation across all wallets
│── wallet_data.csv     # Shared sample dataset (fallback if a user hasn't uploaded their own)
│── generate_data.py    # Script used to generate wallet_data.csv (optional, not needed to run the app)
│── requirements.txt    # Python dependencies
│── README.md           # This file

# Created automatically the first time the app runs:
│── users.json          # Registered accounts (usernames + hashed passwords)
│── user_data/           # Each user's uploaded dataset, one CSV per username
```

---

## 3. Features

### Sidebar
- Select a single wallet, or switch to "All Wallets" view
- Live display of the AI model's test accuracy

### Dashboard Metrics
- Total Trades
- Winning Trades / Losing Trades
- Win Rate
- Total Profit
- Average Profit per Trade

### Charts (Plotly)
- **Profit by Coin** — bar chart
- **Cumulative Profit Over Time** — line chart
- **Winning vs Losing Trades** — pie chart

### Wallet Comparison Table
Every wallet is scored and ranked with:
- Total Profit
- Win Rate
- Wallet Score (0–100)
- Recommendation (Excellent / Good / Average / Risky)

### Wallet Insights
- Best & worst performing coin
- Highest single profit trade
- Average profit per trade
- Wallet consistency (based on win rate)

### AI-Generated Insights (cross-wallet)
Automatically detected patterns scanned across ALL wallets at once —
coin concentration, weak coins, top consistency, and trading efficiency.
Filterable by wallet. See section 6a below for details.

### AI Trade Recommendation
A simple form where you pick a coin, buy price, quantity, and wallet, and the
model predicts whether the trade is **✅ Likely Profitable** or
**❌ Less Likely Profitable**, along with a confidence percentage.

---

## 4. How Profit Is Calculated

For every trade:

```
Profit            = (Sell_Price - Buy_Price) × Quantity
Profit_Percentage = (Sell_Price - Buy_Price) / Buy_Price × 100
Trade_Result       = "Winning Trade" if Profit > 0 else "Losing Trade"
```

These are calculated automatically for every row in `wallet_data.csv` when the
app loads the data (see `load_data()` in `app.py`).

---

## 5. How the Wallet Score Is Calculated

Each wallet is scored out of **100** using a simple rule-based system:

| Condition                  | Points |
|-----------------------------|--------|
| Win Rate > 70%               | +40    |
| Total Profit > 1000          | +30    |
| Average Profit per Trade > 20| +20    |
| Number of Trades > 20        | +10    |

The final score maps to a recommendation:

| Score Range | Recommendation      |
|-------------|----------------------|
| 80 – 100    | 🌟 Excellent Wallet   |
| 60 – 79     | ✅ Good Wallet        |
| 40 – 59     | ⚖️ Average Wallet     |
| 0 – 39      | ⚠️ Risky Wallet       |

This logic lives in `score_wallet()` and `recommendation_from_score()` inside
`app.py`.

---

## 6. How the AI Recommendation Works

The AI feature is intentionally simple — it demonstrates the *workflow* of a
basic supervised learning model, not a real trading strategy.

**Steps (all in `model.py`):**

1. **Target creation** — For every historical trade, we create a label:
   `Profitable_Trade = 1` if `Profit > 0`, else `0`.

2. **Feature engineering** — For each trade we build 5 features:
   - `Coin` (encoded as a number using `LabelEncoder`)
   - `Buy_Price`
   - `Quantity`
   - `Wallet_Win_Rate` — that wallet's historical win rate
   - `Wallet_Avg_Profit` — that wallet's historical average profit

3. **Train/test split** — 80% of trades are used for training, 20% are held
   out to measure accuracy (`train_test_split`, `random_state=42`).

4. **Model training** — A `RandomForestClassifier` (100 trees, max depth 5)
   is trained on the training set.

5. **Prediction** — When you fill in the "AI Trade Recommendation" form, the
   app builds a single-row feature vector (using the selected wallet's
   historical win rate & average profit) and calls `model.predict()` /
   `model.predict_proba()` to get a label and confidence score.

The model accuracy is displayed live in the sidebar.

### Model Used
- **Algorithm:** `RandomForestClassifier` (scikit-learn)
- **Why Random Forest?** It's simple to train, doesn't need feature scaling,
  handles small tabular datasets well, and is easy for beginners to reason
  about compared to deep learning models.

**Explicitly avoided (by design):** deep learning, LSTMs, real-time price
prediction, and any live exchange/blockchain API integration.

---

## 6a. AI-Generated Insights (cross-wallet pattern detection)

Separate from the RandomForest model, the **"🧠 AI-Generated Insights"**
section (`insights_engine.py`) scans the *entire* dataset across *all*
wallets at once and automatically writes out plain-English observations,
for example:

- "Wallet W001 earns 70% of its profits from BTC." *(coin concentration)*
- "Wallet W003 performs poorly in SOL." *(weak coin)*
- "Wallet W005 has the highest consistency." *(best win rate overall)*
- "Wallet W002 trades less but earns more per trade." *(efficiency)*

**⚠️ How this differs from the RandomForest model:** this is **rule-based
statistical analysis**, not a trained machine learning model. It applies a
small checklist of rules to the data (percentage thresholds, min/max
comparisons, dataset-wide averages) rather than learning patterns from
historical examples. It's a different, complementary kind of automation —
more like an automated analyst applying a checklist than a predictive model.

**The 4 rules it checks, for every wallet:**

| Insight Type | Rule |
|---|---|
| 🎯 Coin Concentration | Flags a wallet if ≥50% of its total profit comes from a single coin |
| ⚠️ Weak Coin | Flags a wallet's worst-performing coin, if that coin is a net loss (needs ≥3 trades in that coin to qualify) |
| 🏅 Consistency | Identifies the single wallet with the highest win rate across the whole dataset |
| ⚡ Efficiency | Flags wallets that trade less often than average, but earn more profit per trade than average |

You can filter these insights to a single wallet using the dropdown in that
section, or view all of them at once. Thresholds (like the 50% concentration
cutoff) are defined as constants at the top of `insights_engine.py` if you
want to make the rules stricter or looser.

---

## 7. Dataset Description

`wallet_data.csv` contains **150 simulated trades** across **8 wallets**
(`W001`–`W008`), generated with `generate_data.py` using a fixed random seed
for reproducibility.

| Column      | Description                                  |
|-------------|-----------------------------------------------|
| Wallet_ID   | Unique wallet identifier (W001–W008)           |
| Trade_Date  | Date the trade occurred                        |
| Coin        | Coin traded (BTC, ETH, SOL, BNB, ADA, XRP, DOGE, MATIC) |
| Buy_Price   | Price per unit at purchase                     |
| Sell_Price  | Price per unit at sale                         |
| Quantity    | Amount of the coin traded                      |

Each wallet was generated with a different underlying "skill level" so that
some wallets are consistently profitable, some are average, and some are
risky — this makes the dashboard, scoring, and AI predictions more realistic
and easier to demo.

---

## 8. How to Run the Project

1. **Clone / download** this folder.

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

4. Your browser will open automatically at `http://localhost:8501`.

5. You'll be asked to log in first — see **8a. Login** below.

> Note: `wallet_data.csv` is already included, so you don't need to run
> `generate_data.py`. That script is only provided for reference, in case you
> want to regenerate or expand the sample dataset.

---

## 8a. Accounts & Your Own Dataset

Instead of fixed accounts, this version lets **up to 3 people register their
own username/password** directly on the login page, and each registered
user can **upload their own trade CSV** to analyze instead of the shared
sample data.

### Registering
1. On the login screen, open the **"📝 Register"** tab.
2. Choose a username and password (password must be at least 6 characters).
3. Once registered, switch to the **"🔑 Log In"** tab to sign in.
4. Registration automatically closes once 3 accounts exist. To allow more,
   raise `MAX_USERS` in `auth.py`.

Accounts are stored in `users.json` (created automatically), with passwords
saved as SHA-256 hashes — never in plain text.

### Uploading your own dataset
Once logged in, open the **"📁 Your Dataset"** panel in the sidebar:
- Upload a CSV with these required columns: `Wallet_ID, Trade_Date, Coin,
  Buy_Price, Sell_Price, Quantity`
- The file is validated on upload (checked for the required columns and
  at least one row) before being saved
- Your dataset is stored privately under `user_data/<your_username>_wallet_data.csv`
  and only used for your own dashboard — other users don't see it
- If you haven't uploaded a file yet, the dashboard shows the shared sample
  dataset (`wallet_data.csv`) instead, so there's always something to view
- You can remove your uploaded dataset anytime with the "Remove my dataset"
  button, which reverts you back to the sample data

Each logged-in person gets their own independent Streamlit session, so 3
users can be logged in and viewing their own data at the same time from
different browsers/computers without affecting each other.

**⚠️ Security note:** this is a lightweight, learning-project-level auth
system (SHA-256 hashing, no salting, a plain JSON file for storage) — not
production-grade security. Don't reuse this pattern for an app handling
sensitive real data.

---

## 9. Future Improvements

- Add real historical price data via a public crypto API (with clear
  disclaimers) instead of simulated prices
- Add user-uploaded CSV support so people can analyze their own trade history
- Add more advanced features to the AI model (e.g. holding duration, market
  volatility) and compare multiple model types
- Add authentication and persistent storage (e.g. a database) for multi-user
  support
- Add export functionality (PDF/Excel reports) for wallet summaries
- Add hyperparameter tuning and cross-validation to improve model reliability
- Add a backtesting module to simulate how the AI recommendation would have
  performed historically

---

## Disclaimer

This project is for **educational purposes only**. It uses simulated data and
a simplified machine learning model. It does not constitute financial advice
and should not be used to make real trading or investment decisions.
