"""
app.py
------
Smart Wallet Analyzer with AI Trade Recommendation
====================================================

A beginner-friendly Streamlit dashboard that simulates the analysis of
cryptocurrency wallets using a small local CSV dataset (no real
blockchain / crypto API calls).

This file is organized top-to-bottom the same way a user reads the page:
    1. Page setup & data loading
    2. Profit calculations
    3. Sidebar (wallet selector)
    4. Dashboard metrics
    5. Charts
    6. Wallet comparison table + scoring
    7. Wallet insights
    8. AI Trade Recommendation section

Everything is heavily commented so a beginner can follow every step.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from model import train_model, predict_trade
from auth import require_login, logout_button
from styles import load_css
from insights_engine import generate_insights
from user_data import (
    get_active_dataset_path,
    has_user_dataset,
    validate_csv,
    save_user_dataset,
    delete_user_dataset,
    REQUIRED_COLUMNS,
)

# ---------------------------------------------------------------------------
# 1. PAGE SETUP
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Smart Wallet Analyzer",
    page_icon="💹",
    layout="wide",
)

# --- Require login before anything else renders ---
require_login()

# --- Custom CSS theme (applies once the user is logged in) ---
load_css("style.css")

st.title("💹 Smart Wallet Analyzer with AI Trade Recommendation")
st.caption(
    "A learning project that simulates crypto wallet analysis using sample data. "
    "This is NOT connected to any real blockchain, exchange, or crypto API."
)

current_user = st.session_state.username
logout_button()
st.sidebar.markdown("---")


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Load the trade data from CSV and calculate profit-related columns."""
    df = pd.read_csv(path, parse_dates=["Trade_Date"])

    # --- Core calculations required by the project spec ---
    df["Profit"] = (df["Sell_Price"] - df["Buy_Price"]) * df["Quantity"]
    df["Profit_Percentage"] = (
        (df["Sell_Price"] - df["Buy_Price"]) / df["Buy_Price"] * 100
    )
    df["Trade_Result"] = df["Profit"].apply(
        lambda p: "Winning Trade" if p > 0 else "Losing Trade"
    )

    return df


@st.cache_resource
def load_model(df: pd.DataFrame):
    """Train the RandomForest model once and cache it for the session."""
    return train_model(df)


# ---------------------------------------------------------------------------
# 1b. PER-USER DATASET UPLOAD
# ---------------------------------------------------------------------------
# Each logged-in user can upload their own trade CSV. If they haven't
# uploaded one yet, the app falls back to the shared sample dataset so
# the dashboard always has something to show.

with st.sidebar.expander("📁 Your Dataset", expanded=not has_user_dataset(current_user)):
    if has_user_dataset(current_user):
        st.success("Using your uploaded dataset.")
        if st.button("🗑️ Remove my dataset (use sample data instead)", use_container_width=True):
            delete_user_dataset(current_user)
            st.cache_data.clear()
            st.rerun()
    else:
        st.info("You're currently viewing the shared sample dataset.")

    st.caption(f"Required columns: {', '.join(REQUIRED_COLUMNS)}")

    uploaded_file = st.file_uploader("Upload your trades CSV", type=["csv"], key="dataset_uploader")

    if uploaded_file is not None:
        is_valid, message, parsed_df = validate_csv(uploaded_file)
        if is_valid:
            save_user_dataset(current_user, parsed_df)
            st.cache_data.clear()
            st.success(f"Dataset uploaded! {message}")
            st.rerun()
        else:
            st.error(message)

active_dataset_path = get_active_dataset_path(current_user)
df = load_data(active_dataset_path)
model, accuracy, coin_encoder, wallet_stats = load_model(df)


# ---------------------------------------------------------------------------
# 2. SIDEBAR - WALLET SELECTION
# ---------------------------------------------------------------------------

st.sidebar.header("🔎 Wallet Selection")

all_wallets = sorted(df["Wallet_ID"].unique())
view_option = st.sidebar.radio("View mode", ["Single Wallet", "All Wallets"])

if view_option == "Single Wallet":
    selected_wallet = st.sidebar.selectbox("Select a Wallet", all_wallets)
    filtered_df = df[df["Wallet_ID"] == selected_wallet]
else:
    selected_wallet = None
    filtered_df = df.copy()

st.sidebar.markdown("---")
st.sidebar.metric("🤖 AI Model Accuracy", f"{accuracy * 100:.1f}%")
st.sidebar.caption(
    "Accuracy = how often the model was correct on unseen test trades "
    "(measured once, overall). This is different from the 'Confidence' % "
    "shown after each individual prediction below, which reflects how sure "
    "the model is about that one trade, not whether it will actually be right. "
    "This is a small demo dataset, so accuracy is illustrative only."
)


# ---------------------------------------------------------------------------
# 3. DASHBOARD METRICS
# ---------------------------------------------------------------------------

st.subheader(
    f"📊 Dashboard Metrics — {selected_wallet if selected_wallet else 'All Wallets'}"
)

total_trades = len(filtered_df)
winning_trades = (filtered_df["Profit"] > 0).sum()
losing_trades = (filtered_df["Profit"] <= 0).sum()
win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
total_profit = filtered_df["Profit"].sum()
avg_profit = filtered_df["Profit"].mean() if total_trades > 0 else 0

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Total Trades", f"{total_trades}")
col2.metric("Winning Trades", f"{winning_trades}")
col3.metric("Losing Trades", f"{losing_trades}")
col4.metric("Win Rate", f"{win_rate:.1f}%")
col5.metric("Total Profit", f"${total_profit:,.2f}")
col6.metric("Avg Profit / Trade", f"${avg_profit:,.2f}")

st.markdown("---")


# ---------------------------------------------------------------------------
# 4. CHARTS
# ---------------------------------------------------------------------------

st.subheader("📈 Charts")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    # --- Profit by Coin (Bar Chart) ---
    profit_by_coin = (
        filtered_df.groupby("Coin")["Profit"].sum().reset_index().sort_values("Profit")
    )
    fig_bar = px.bar(
        profit_by_coin,
        x="Coin",
        y="Profit",
        color="Profit",
        color_continuous_scale=["#e74c3c", "#2ecc71"],
        title="Profit by Coin",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with chart_col2:
    # --- Winning vs Losing Trades (Pie Chart) ---
    result_counts = filtered_df["Trade_Result"].value_counts().reset_index()
    result_counts.columns = ["Trade_Result", "Count"]
    fig_pie = px.pie(
        result_counts,
        names="Trade_Result",
        values="Count",
        color="Trade_Result",
        color_discrete_map={"Winning Trade": "#2ecc71", "Losing Trade": "#e74c3c"},
        title="Winning vs Losing Trades",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# --- Profit Over Time (Line Chart) ---
profit_over_time = (
    filtered_df.sort_values("Trade_Date")
    .groupby("Trade_Date")["Profit"]
    .sum()
    .cumsum()
    .reset_index()
)
profit_over_time.rename(columns={"Profit": "Cumulative_Profit"}, inplace=True)

fig_line = px.line(
    profit_over_time,
    x="Trade_Date",
    y="Cumulative_Profit",
    title="Cumulative Profit Over Time",
    markers=True,
)
st.plotly_chart(fig_line, use_container_width=True)

st.markdown("---")


# ---------------------------------------------------------------------------
# 4b. COIN PROFIT / LOSS BREAKDOWN (button-triggered, per wallet + all wallets)
# ---------------------------------------------------------------------------

st.subheader("🪙 Coin Profit/Loss Breakdown")
st.caption(
    "See what percentage each coin contributes to total profit and total loss - "
    "one view for all wallets combined, plus a separate view for each individual wallet."
)


def render_coin_breakdown_chart(data: pd.DataFrame, chart_key: str):
    """Build and display the coin profit/loss % breakdown chart for a given dataset."""
    coin_profit_totals = data.groupby("Coin")["Profit"].sum()

    profit_coins = coin_profit_totals[coin_profit_totals > 0]
    loss_coins = coin_profit_totals[coin_profit_totals <= 0]

    total_profit_sum = profit_coins.sum()
    total_loss_sum = loss_coins.abs().sum()

    breakdown_rows = []
    for coin, amount in profit_coins.items():
        pct = (amount / total_profit_sum * 100) if total_profit_sum > 0 else 0
        breakdown_rows.append({
            "Coin": coin,
            "Percentage": round(pct, 1),
            "Category": "Profit Contribution",
            "Amount": round(amount, 2),
        })
    for coin, amount in loss_coins.items():
        pct = (abs(amount) / total_loss_sum * 100) if total_loss_sum > 0 else 0
        breakdown_rows.append({
            "Coin": coin,
            "Percentage": round(pct, 1),
            "Category": "Loss Contribution",
            "Amount": round(amount, 2),
        })

    if len(breakdown_rows) == 0:
        st.info("No trade data available to build this breakdown.")
        return

    breakdown_df = pd.DataFrame(breakdown_rows).sort_values(
        ["Category", "Percentage"], ascending=[True, False]
    )

    fig_breakdown = px.bar(
        breakdown_df,
        x="Percentage",
        y="Coin",
        color="Category",
        orientation="h",
        color_discrete_map={
            "Profit Contribution": "#2ecc71",
            "Loss Contribution": "#e74c3c",
        },
        text="Percentage",
        hover_data={"Amount": True, "Percentage": True},
        title="% Share of Total Profit / Total Loss by Coin",
    )
    fig_breakdown.update_traces(texttemplate="%{text}%", textposition="outside")
    fig_breakdown.update_layout(xaxis_title="% Share", yaxis_title="Coin")
    st.plotly_chart(fig_breakdown, use_container_width=True, key=chart_key)

    st.caption(
        "Profit Contribution % = this coin's share of total profit (among "
        "profitable coins only). Loss Contribution % = this coin's share of "
        "total loss (among unprofitable coins only). Each group adds up to 100% on its own."
    )


if "show_coin_breakdown" not in st.session_state:
    st.session_state.show_coin_breakdown = False

breakdown_col1, breakdown_col2 = st.columns([1, 3])
with breakdown_col1:
    if st.button("📊 Show Coin Breakdown", use_container_width=True):
        st.session_state.show_coin_breakdown = not st.session_state.show_coin_breakdown

if st.session_state.show_coin_breakdown:
    breakdown_tabs = st.tabs(["🌐 All Wallets"] + [f"👛 {w}" for w in all_wallets])

    # --- Tab 1: All wallets combined ---
    with breakdown_tabs[0]:
        render_coin_breakdown_chart(df, chart_key="breakdown_all_wallets")

    # --- One tab per individual wallet ---
    for i, wallet in enumerate(all_wallets, start=1):
        with breakdown_tabs[i]:
            wallet_df = df[df["Wallet_ID"] == wallet]
            render_coin_breakdown_chart(wallet_df, chart_key=f"breakdown_{wallet}")

st.markdown("---")


# ---------------------------------------------------------------------------
# 5. WALLET COMPARISON TABLE + SCORING SYSTEM
# ---------------------------------------------------------------------------

st.subheader("🏆 Wallet Comparison Table")


def score_wallet(trades: int, win_rate_pct: float, total_profit_val: float, avg_profit_val: float) -> int:
    """
    Simple 0-100 scoring system based on the spec's rules:
        Win Rate > 70%      -> +40
        Total Profit > 1000 -> +30
        Avg Profit > 20     -> +20
        Trades > 20         -> +10
    """
    score = 0
    if win_rate_pct > 70:
        score += 40
    if total_profit_val > 1000:
        score += 30
    if avg_profit_val > 20:
        score += 20
    if trades > 20:
        score += 10
    return score


def recommendation_from_score(score: int) -> str:
    """Map a 0-100 score to a human-readable recommendation label."""
    if score >= 80:
        return "🌟 Excellent Wallet"
    elif score >= 60:
        return "✅ Good Wallet"
    elif score >= 40:
        return "⚖️ Average Wallet"
    else:
        return "⚠️ Risky Wallet"


# Build one summary row per wallet
comparison_rows = []
for wallet_id in all_wallets:
    w_df = df[df["Wallet_ID"] == wallet_id]
    w_trades = len(w_df)
    w_wins = (w_df["Profit"] > 0).sum()
    w_win_rate = (w_wins / w_trades * 100) if w_trades > 0 else 0
    w_total_profit = w_df["Profit"].sum()
    w_avg_profit = w_df["Profit"].mean() if w_trades > 0 else 0

    w_score = score_wallet(w_trades, w_win_rate, w_total_profit, w_avg_profit)
    w_recommendation = recommendation_from_score(w_score)

    comparison_rows.append(
        {
            "Wallet ID": wallet_id,
            "Total Profit": round(w_total_profit, 2),
            "Win Rate (%)": round(w_win_rate, 1),
            "Wallet Score": w_score,
            "Recommendation": w_recommendation,
        }
    )

comparison_df = pd.DataFrame(comparison_rows).sort_values("Wallet Score", ascending=False)
st.dataframe(comparison_df, use_container_width=True, hide_index=True)

st.markdown("---")


# ---------------------------------------------------------------------------
# 5b. AI-GENERATED INSIGHTS (cross-wallet pattern detection)
# ---------------------------------------------------------------------------

st.subheader("🧠 AI-Generated Insights")
st.caption(
    "Automatically detected patterns across ALL wallets - coin concentration, "
    "weak spots, consistency, and trading efficiency. This uses rule-based "
    "statistical analysis over the whole dataset, not the RandomForest model "
    "used in the AI Trade Recommendation section below."
)

all_insights = generate_insights(df)

if len(all_insights) == 0:
    st.info("Not enough data yet to generate insights.")
else:
    # Let the user filter insights to a specific wallet, or see everything
    insight_wallet_filter = st.selectbox(
        "Filter insights by wallet",
        ["All Wallets"] + all_wallets,
        key="insight_wallet_filter",
    )

    if insight_wallet_filter != "All Wallets":
        shown_insights = [i for i in all_insights if i["wallet"] == insight_wallet_filter]
    else:
        shown_insights = all_insights

    if len(shown_insights) == 0:
        st.info(f"No notable patterns detected for {insight_wallet_filter}.")
    else:
        for insight in shown_insights:
            st.markdown(f"{insight['icon']} **[{insight['type']}]** {insight['text']}")

st.markdown("---")


# ---------------------------------------------------------------------------
# 6. WALLET INSIGHTS
# ---------------------------------------------------------------------------

st.subheader(
    f"💡 Wallet Insights — {selected_wallet if selected_wallet else 'All Wallets'}"
)

if total_trades > 0:
    coin_profit = filtered_df.groupby("Coin")["Profit"].sum()
    best_coin = coin_profit.idxmax()
    worst_coin = coin_profit.idxmin()

    highest_profit_trade = filtered_df.loc[filtered_df["Profit"].idxmax()]

    # Simple "consistency" label based on win rate
    if win_rate >= 70:
        consistency = "Very Consistent"
    elif win_rate >= 50:
        consistency = "Moderately Consistent"
    else:
        consistency = "Inconsistent"

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:
        st.write(f"**Best Performing Coin:** {best_coin} (${coin_profit[best_coin]:,.2f} total profit)")
        st.write(f"**Worst Performing Coin:** {worst_coin} (${coin_profit[worst_coin]:,.2f} total profit)")
        st.write(f"**Average Profit per Trade:** ${avg_profit:,.2f}")

    with insight_col2:
        st.write(
            f"**Highest Profit Trade:** {highest_profit_trade['Coin']} on "
            f"{highest_profit_trade['Trade_Date'].date()} "
            f"(${highest_profit_trade['Profit']:,.2f})"
        )
        st.write(f"**Wallet Consistency:** {consistency} (Win Rate: {win_rate:.1f}%)")
else:
    st.info("No trades available for this selection.")

st.markdown("---")


# ---------------------------------------------------------------------------
# 6b. VIEW TRADES ON DEMAND (Profitable / Losing)
# ---------------------------------------------------------------------------

st.subheader("🔍 View Trades")
st.caption("Click a button below to reveal that group of trades. Nothing is shown until you ask for it.")

display_cols = ["Wallet_ID", "Trade_Date", "Coin", "Buy_Price", "Sell_Price", "Quantity", "Profit", "Profit_Percentage"]

# Buttons live in columns side by side
btn_col1, btn_col2, btn_col3 = st.columns(3)
show_profitable = btn_col1.button("✅ Show Profitable Trades", use_container_width=True)
show_losing = btn_col2.button("❌ Show Losing Trades", use_container_width=True)
hide_all = btn_col3.button("🙈 Hide Trades", use_container_width=True)

# Streamlit reruns the whole script on every click, so we use session_state
# to "remember" which table (if any) should stay open after the click.
if "trades_view" not in st.session_state:
    st.session_state.trades_view = None

if show_profitable:
    st.session_state.trades_view = "profitable"
if show_losing:
    st.session_state.trades_view = "losing"
if hide_all:
    st.session_state.trades_view = None

if st.session_state.trades_view == "profitable":
    profitable_df = filtered_df[filtered_df["Profit"] > 0].sort_values("Profit", ascending=False)
    st.markdown(f"**✅ Profitable Trades** — {len(profitable_df)} of {total_trades} trades")

    if len(profitable_df) > 0:
        st.dataframe(
            profitable_df[display_cols].style.format(
                {"Buy_Price": "${:.4f}", "Sell_Price": "${:.4f}", "Profit": "${:.2f}", "Profit_Percentage": "{:.2f}%"}
            ),
            use_container_width=True,
            hide_index=True,
        )
        csv_bytes = profitable_df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Profitable Trades as CSV",
            data=csv_bytes,
            file_name="profitable_trades.csv",
            mime="text/csv",
        )
    else:
        st.info("No profitable trades found for this selection.")

elif st.session_state.trades_view == "losing":
    losing_df = filtered_df[filtered_df["Profit"] <= 0].sort_values("Profit")
    st.markdown(f"**❌ Losing Trades** — {len(losing_df)} of {total_trades} trades")

    if len(losing_df) > 0:
        st.dataframe(
            losing_df[display_cols].style.format(
                {"Buy_Price": "${:.4f}", "Sell_Price": "${:.4f}", "Profit": "${:.2f}", "Profit_Percentage": "{:.2f}%"}
            ),
            use_container_width=True,
            hide_index=True,
        )
        csv_bytes = losing_df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Losing Trades as CSV",
            data=csv_bytes,
            file_name="losing_trades.csv",
            mime="text/csv",
        )
    else:
        st.info("No losing trades found for this selection.")

st.markdown("---")


# ---------------------------------------------------------------------------
# 7. AI TRADE RECOMMENDATION
# ---------------------------------------------------------------------------

st.subheader("🤖 AI Trade Recommendation")
st.caption(
    "This section uses a simple RandomForestClassifier trained on historical "
    "trade data to estimate whether a hypothetical trade is likely to be profitable."
)

with st.form("ai_prediction_form"):
    form_col1, form_col2, form_col3, form_col4 = st.columns(4)

    with form_col1:
        input_coin = st.selectbox("Coin", sorted(df["Coin"].unique()))
    with form_col2:
        input_buy_price = st.number_input("Buy Price", min_value=0.0, value=100.0, step=1.0)
    with form_col3:
        input_quantity = st.number_input("Quantity", min_value=0.0, value=10.0, step=1.0)
    with form_col4:
        input_wallet = st.selectbox("Wallet", all_wallets)

    submitted = st.form_submit_button("Get AI Recommendation")

if submitted:
    label, confidence = predict_trade(
        model, coin_encoder, wallet_stats,
        coin=input_coin,
        buy_price=input_buy_price,
        quantity=input_quantity,
        wallet_id=input_wallet,
    )

    icon = "✅" if label == "Likely Profitable" else "❌"

    st.markdown("### AI Recommendation")
    r1, r2, r3 = st.columns(3)
    r1.metric("Coin", input_coin)
    r2.metric("Prediction", f"{icon} {label}")
    r3.metric("Confidence", f"{confidence}%")

    st.info(
        "This prediction is based on historical trading patterns of similar "
        "wallets and previous profitable trades. This is a learning "
        "demonstration and should not be considered financial advice."
    )

st.markdown("---")
st.caption(
    "Smart Wallet Analyzer • Educational / demo project only • "
    "No real blockchain, exchange, or financial data is used."
)
