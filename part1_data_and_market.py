"""
PART 1 — DATA & MARKET INPUTS  (SunPharma Derivatives Project)
================================================================
This is the ONLY script that talks to Yahoo Finance and the ONLY place
market inputs (spot, volatility, risk-free rate, option chain) are
defined. Every other script (Part 2: pricing/Greeks/vol surface,
Part 3: portfolio/hedging/VaR) reads `market_inputs.json` produced here
instead of hardcoding numbers — so if you rerun this file with fresh
data, everything downstream moves with it automatically.

SECTIONS
--------
1. Download historical prices (yfinance)
2. Compute log returns & statistics (mean, std, annualized vol, skew, kurtosis)
3. Save price series + summary stats + plots
4. Fetch the live option chain (yfinance) for the nearest available expiries
5. Export market_inputs.json — the single source of truth for Parts 2 & 3

NOTE ON OPTION CHAINS FOR NSE STOCKS
-------------------------------------
Yahoo Finance's option-chain endpoint (`Ticker.options` / `Ticker.option_chain`)
is populated mainly for US-listed names. Many NSE tickers (e.g. SUNPHARMA.NS)
either return an empty expiry list or raise an error, because Yahoo doesn't
source NSE F&O chain data. This script tries the yfinance route first and
tells you clearly if it comes back empty — in that case, pull the chain
manually from the NSE option-chain page and paste it into the
`MANUAL_OPTION_CHAIN` list below (same schema yfinance would have given you).
"""

import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from datetime import datetime, timezone

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
TICKER = "SUNPHARMA.NS"
HIST_PERIOD = "3mo"          # ~60-65 trading days, matches original methodology
RISK_FREE_RATE = 0.065       # update to current short-term G-Sec/T-bill yield if you have one
OUT_DIR = "./sunpharma_output"
os.makedirs(OUT_DIR, exist_ok=True)

# Fallback option chain — only used if the yfinance fetch below returns nothing.
# Fill this in with real NSE data (Type, Strike, LTP, Volume, Expiry as 'YYYY-MM-DD')
# if you need to pull it manually.
MANUAL_OPTION_CHAIN = []

# ------------------------------------------------------------------
# SECTION 1: Download historical prices
# ------------------------------------------------------------------
print("=" * 100)
print(f"SECTION 1: Downloading {HIST_PERIOD} of daily prices for {TICKER}")
print("=" * 100)

df = yf.download(TICKER, period=HIST_PERIOD, interval="1d", auto_adjust=False, actions=False)

if df.empty:
    raise RuntimeError(
        f"No data returned for {TICKER}. Check your network connection / ticker symbol. "
        "If you're running this in a sandboxed environment without access to "
        "query1.finance.yahoo.com / query2.finance.yahoo.com, run it locally instead."
    )

df.index = pd.to_datetime(df.index)
df = df.sort_index()
df["Price"] = df["Close"]

# ------------------------------------------------------------------
# SECTION 2: Log returns & statistics
# ------------------------------------------------------------------
print("\n" + "=" * 100)
print("SECTION 2: Computing log returns and statistics")
print("=" * 100)

df["Log_Returns"] = np.log(df["Price"] / df["Price"].shift(1))
df = df.dropna()

daily_mean = float(df["Log_Returns"].mean())
daily_std = float(df["Log_Returns"].std())
annual_vol = float(daily_std * np.sqrt(252))
skew_val = float(skew(df["Log_Returns"]))
kurt_val = float(kurtosis(df["Log_Returns"], fisher=False))
n_obs = int(df.shape[0])

latest_date = df.index[-1]
spot_price = float(df["Price"].iloc[-1])

summary = pd.DataFrame({
    "Metric": ["Daily Mean Return", "Daily Std Dev", "Annualized Volatility",
               "Skewness", "Kurtosis", "Observations", "Spot Price", "As-of Date"],
    "Value": [daily_mean, daily_std, annual_vol, skew_val, kurt_val, n_obs,
              spot_price, latest_date.strftime("%Y-%m-%d")]
})

print(summary.to_string(index=False))

if n_obs < 40:
    print(f"\nNOTE: only {n_obs} observations — volatility/skew/kurtosis estimates "
          f"from this sample size carry meaningful statistical noise, especially "
          f"for higher moments (skew/kurtosis). Treat them as indicative, not precise.")

# ------------------------------------------------------------------
# SECTION 3: Save price series, summary, plots
# ------------------------------------------------------------------
df.to_csv(f"{OUT_DIR}/sunpharma_prices.csv")
summary.to_csv(f"{OUT_DIR}/sunpharma_summary.csv", index=False)

plt.figure(figsize=(12, 4))
plt.plot(df.index, df["Log_Returns"])
plt.title(f"{TICKER} – Daily Log Returns ({HIST_PERIOD})")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/log_returns_ts.png")
plt.close()

plt.figure(figsize=(8, 4))
plt.hist(df["Log_Returns"], bins=30)
plt.title(f"{TICKER} – Histogram of Daily Log Returns")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/log_returns_hist.png")
plt.close()

print(f"\nSaved price series, summary, and plots to {OUT_DIR}/")

# ------------------------------------------------------------------
# SECTION 4: Fetch live option chain (yfinance)
# ------------------------------------------------------------------
print("\n" + "=" * 100)
print("SECTION 4: Fetching live option chain")
print("=" * 100)

ticker_obj = yf.Ticker(TICKER)
option_chain_rows = []

try:
    expiries = ticker_obj.options  # tuple of expiry date strings, e.g. ('2026-09-25', ...)
except Exception as e:
    expiries = ()
    print(f"yfinance option-chain lookup raised an error: {e}")

if not expiries:
    print(f"yfinance returned NO option expiries for {TICKER}.")
    print("This is expected for many NSE-listed tickers — Yahoo Finance does not "
          "source NSE F&O chain data for most Indian stocks.")
    if MANUAL_OPTION_CHAIN:
        print(f"Using MANUAL_OPTION_CHAIN with {len(MANUAL_OPTION_CHAIN)} contracts instead.")
        option_chain_rows = MANUAL_OPTION_CHAIN
    else:
        print("MANUAL_OPTION_CHAIN is empty. Pull real strikes/LTPs/volumes/expiries "
              "from the NSE option-chain page (nseindia.com) and paste them into "
              "MANUAL_OPTION_CHAIN at the top of this file, then rerun.")
else:
    print(f"Found {len(expiries)} expiries: {expiries}")
    # Pull the two nearest expiries so a volatility surface (Part 2) has two
    # maturities to interpolate across, same as the original project's design.
    for expiry in expiries[:2]:
        chain = ticker_obj.option_chain(expiry)
        for _, row in chain.calls.iterrows():
            option_chain_rows.append({
                "Type": "Call", "Strike": float(row["strike"]),
                "LTP": float(row["lastPrice"]), "Volume": int(row["volume"] or 0),
                "Expiry": expiry
            })
        for _, row in chain.puts.iterrows():
            option_chain_rows.append({
                "Type": "Put", "Strike": float(row["strike"]),
                "LTP": float(row["lastPrice"]), "Volume": int(row["volume"] or 0),
                "Expiry": expiry
            })
    print(f"Collected {len(option_chain_rows)} option contracts across {min(2, len(expiries))} expiries.")

option_chain_df = pd.DataFrame(option_chain_rows)
option_chain_df.to_csv(f"{OUT_DIR}/sunpharma_option_chain.csv", index=False)
print(f"Option chain saved to {OUT_DIR}/sunpharma_option_chain.csv "
      f"({len(option_chain_df)} rows — 0 rows means you need to fill in MANUAL_OPTION_CHAIN)")

# ------------------------------------------------------------------
# SECTION 4b: Cross-check spot against the option chain (put-call parity)
# ------------------------------------------------------------------
# The downloaded price series and the option chain can come from different
# moments (e.g. yesterday's close vs. today's live quotes). If they disagree,
# every downstream IV/Greek calculation gets built on a stale reference
# price. Rather than silently pricing off whichever number happened to load
# first, infer the spot the market is actually quoting options against and
# flag any material gap.
if not option_chain_df.empty:
    print("\n" + "-" * 100)
    print("Cross-checking spot price against option chain (put-call parity)")
    print("-" * 100)

    nearest_expiry = sorted(option_chain_df["Expiry"].unique())[0]
    chain_near = option_chain_df[option_chain_df["Expiry"] == nearest_expiry]
    T_near = (datetime.strptime(nearest_expiry, "%Y-%m-%d") - latest_date.to_pydatetime()).days / 365

    implied_spots = []
    if T_near > 0:
        common_strikes = (set(chain_near[chain_near.Type == "Call"]["Strike"])
                           & set(chain_near[chain_near.Type == "Put"]["Strike"]))
        for K in sorted(common_strikes):
            call_ltp = chain_near[(chain_near.Type == "Call") & (chain_near.Strike == K)]["LTP"].iloc[0]
            put_ltp = chain_near[(chain_near.Type == "Put") & (chain_near.Strike == K)]["LTP"].iloc[0]
            implied_spots.append((call_ltp - put_ltp) + K * np.exp(-RISK_FREE_RATE * T_near))

    if implied_spots:
        implied_spot = float(np.mean(implied_spots))
        gap_pct = abs(implied_spot - spot_price) / spot_price * 100
        print(f"Downloaded spot (latest close): {spot_price:.2f}")
        print(f"Option-chain-implied spot ({len(implied_spots)} strikes, {nearest_expiry}): {implied_spot:.2f}")
        print(f"Gap: {gap_pct:.2f}%")

        if gap_pct > 1.0:
            print(f"\nWARNING: >{1.0}% gap between the downloaded close and the "
                  f"option-chain-implied spot. This usually means the price series "
                  f"and the option chain were pulled on different dates/times. "
                  f"Using the option-chain-implied spot ({implied_spot:.2f}) for "
                  f"market_inputs.json so pricing/Greeks/VaR stay internally "
                  f"consistent with the contracts actually being priced.")
            spot_price = implied_spot
    else:
        print("Not enough overlapping call/put strikes to cross-check spot — skipping.")

# ------------------------------------------------------------------
# SECTION 5: Export market_inputs.json — single source of truth
# ------------------------------------------------------------------
print("\n" + "=" * 100)
print("SECTION 5: Writing market_inputs.json")
print("=" * 100)

# Strikes for the pricing grid in Part 2: 5 strikes centered on spot,
# spaced the same way as the original TechM project (~±2%, ±5%).
strike_offsets = [-0.05, -0.02, 0.0, 0.02, 0.05]
strikes_for_analysis = [round(spot_price * (1 + o), 3) for o in strike_offsets]

market_inputs = {
    "ticker": TICKER,
    "valuation_date": latest_date.strftime("%Y-%m-%d"),
    "spot": spot_price,
    "risk_free_rate": RISK_FREE_RATE,
    "historical_volatility": annual_vol,
    "daily_mean_return": daily_mean,
    "daily_std_dev": daily_std,
    "skewness": skew_val,
    "kurtosis": kurt_val,
    "observations": n_obs,
    "strikes_for_analysis": strikes_for_analysis,
    "maturities_days": [30, 60, 90],
    "generated_at": datetime.now(timezone.utc).isoformat()
}

with open(f"{OUT_DIR}/market_inputs.json", "w") as f:
    json.dump(market_inputs, f, indent=2)

print(json.dumps(market_inputs, indent=2))
print(f"\nmarket_inputs.json written to {OUT_DIR}/ — Parts 2 & 3 will read this file directly.")
print("=" * 100)
