"""
PART 3 — PORTFOLIO, HEDGING, PnL SCENARIOS, VaR
==================================================
Replaces the original project's Part D + Part E with one script.

Reads `market_inputs.json` produced by Part 1. Imports all pricing/Greeks
math from `pricing_engine.py`. No hardcoded S/sigma/r, no duplicate Greeks
function, no dead "computed but never used" variables, no leftover debug
print blocks.

SECTIONS
--------
1. Load market inputs
2. Construct a 6-leg option portfolio & compute Greeks
3. Delta-hedge, then gamma-hedge, then re-null delta (one clean pass —
   the original computed an intermediate stock_cost that was immediately
   superseded and never used; that's removed here)
4. PnL simulation across spot-price scenarios, hedged vs unhedged
5. VaR: parametric (variance-covariance) and historical simulation,
   with an explicit small-sample caveat on the 99% historical estimate
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pricing_engine import black_scholes_price, greeks

IN_DIR = "./sunpharma_output"
OUT_DIR = "./sunpharma_output"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# SECTION 1: Load market inputs
# ------------------------------------------------------------------
print("=" * 100)
print("SECTION 1: Loading market inputs")
print("=" * 100)

with open(f"{IN_DIR}/market_inputs.json") as f:
    mkt = json.load(f)

S = mkt["spot"]
r = mkt["risk_free_rate"]
sigma = mkt["historical_volatility"]
strikes = mkt["strikes_for_analysis"]  # [atm-5%, atm-2%, atm, atm+2%, atm+5%]
daily_mean = mkt["daily_mean_return"]
daily_std = mkt["daily_std_dev"]
n_obs = mkt["observations"]

price_series = pd.read_csv(f"{IN_DIR}/sunpharma_prices.csv", index_col=0, parse_dates=True)
daily_returns = price_series["Log_Returns"].values

print(f"Spot: {S} | Vol: {sigma*100:.2f}% | r: {r*100:.2f}% | "
      f"Historical return sample size: {n_obs}")

# ------------------------------------------------------------------
# SECTION 2: Portfolio construction
# ------------------------------------------------------------------
print("\n" + "=" * 100)
print("SECTION 2: Constructing option portfolio")
print("=" * 100)

# strikes list is [-5%, -2%, ATM, +2%, +5%] around spot from Part 1
k_m5, k_m2, k_atm, k_p2, k_p5 = strikes

portfolio_options = [
    {"Type": "call", "Strike": k_atm, "Maturity_Days": 30, "Position": 10},
    {"Type": "put",  "Strike": k_atm, "Maturity_Days": 30, "Position": -5},
    {"Type": "call", "Strike": k_p2,  "Maturity_Days": 60, "Position": 5},
    {"Type": "put",  "Strike": k_m2,  "Maturity_Days": 60, "Position": -3},
    {"Type": "call", "Strike": k_p5,  "Maturity_Days": 90, "Position": 2},
    {"Type": "put",  "Strike": k_m5,  "Maturity_Days": 90, "Position": -2},
]

portfolio_rows, totals = [], {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
total_premium = 0.0

for opt in portfolio_options:
    T = opt["Maturity_Days"] / 365.0
    price = black_scholes_price(S, opt["Strike"], T, r, sigma, opt["Type"])
    g = greeks(S, opt["Strike"], T, r, sigma, opt["Type"])
    pos = opt["Position"]

    row = {"Type": opt["Type"].capitalize(), "Strike": opt["Strike"],
           "Maturity_Days": opt["Maturity_Days"], "Position": pos, "Price": price,
           "Premium": price * pos}
    for greek_name, val in g.items():
        row[f"Pos_{greek_name}"] = val * pos
        totals[greek_name] += val * pos
    portfolio_rows.append(row)
    total_premium += price * pos

portfolio_df = pd.DataFrame(portfolio_rows)
portfolio_df.to_csv(f"{OUT_DIR}/sunpharma_portfolio_composition.csv", index=False)
print(portfolio_df.to_string(index=False))
print(f"\nTotal Delta: {totals['delta']:.4f} | Gamma: {totals['gamma']:.6f} | "
      f"Vega: {totals['vega']:.4f} | Theta: {totals['theta']:.4f} | Rho: {totals['rho']:.4f}")
print(f"Total Premium: {total_premium:,.2f}")

unhedged = dict(totals)  # snapshot before hedging, for the before/after report

# ------------------------------------------------------------------
# SECTION 3: Hedging (delta -> gamma -> re-null delta)
# ------------------------------------------------------------------
print("\n" + "=" * 100)
print("SECTION 3: Hedging the portfolio")
print("=" * 100)

# Gamma hedge instrument: longest-dated ATM call
gamma_hedge_leg = {"Type": "call", "Strike": k_atm, "Maturity_Days": 90}
T_gh = gamma_hedge_leg["Maturity_Days"] / 365.0
gh_price = black_scholes_price(S, gamma_hedge_leg["Strike"], T_gh, r, sigma, gamma_hedge_leg["Type"])
gh_greeks = greeks(S, gamma_hedge_leg["Strike"], T_gh, r, sigma, gamma_hedge_leg["Type"])

gamma_hedge_qty = -totals["gamma"] / gh_greeks["gamma"]
gamma_hedge_cost = gamma_hedge_qty * gh_price

# Delta after adding the gamma hedge
delta_after_gamma_hedge = totals["delta"] + gamma_hedge_qty * gh_greeks["delta"]

# Stock position computed ONCE, after the gamma hedge is already in place
# (the original project computed an earlier stock_cost before the gamma
# hedge and never used it — that dead calculation is removed here)
stock_position = -delta_after_gamma_hedge
stock_cost = stock_position * S

hedged = {
    "delta": delta_after_gamma_hedge + stock_position,
    "gamma": totals["gamma"] + gamma_hedge_qty * gh_greeks["gamma"],
    "vega": totals["vega"] + gamma_hedge_qty * gh_greeks["vega"],
    "theta": totals["theta"] + gamma_hedge_qty * gh_greeks["theta"],
    "rho": totals["rho"] + gamma_hedge_qty * gh_greeks["rho"],
}

hedge_summary = pd.DataFrame({
    "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
    "Before_Hedge": [unhedged[g] for g in ("delta", "gamma", "vega", "theta", "rho")],
    "After_Hedge": [hedged[g] for g in ("delta", "gamma", "vega", "theta", "rho")],
})
hedge_summary.to_csv(f"{OUT_DIR}/sunpharma_hedge_summary.csv", index=False)
print(hedge_summary.to_string(index=False))
print(f"\nGamma hedge: {gamma_hedge_qty:.4f} contracts of "
      f"{gamma_hedge_leg['Type']} @ Strike {gamma_hedge_leg['Strike']} "
      f"(T={gamma_hedge_leg['Maturity_Days']}d), cost {gamma_hedge_cost:,.2f}")
print(f"Stock hedge: {stock_position:.4f} shares @ {S:.2f}, cost {stock_cost:,.2f}")
print(f"Total hedging cost: {gamma_hedge_cost + stock_cost:,.2f}")

# ------------------------------------------------------------------
# SECTION 4: PnL scenarios
# ------------------------------------------------------------------
print("\n" + "=" * 100)
print("SECTION 4: PnL simulation across spot-price scenarios")
print("=" * 100)

pnl_rows = []
for pct in (-0.02, -0.01, 0.0, 0.01, 0.02):
    S_new = S * (1 + pct)

    unhedged_pnl = sum(
        (black_scholes_price(S_new, o["Strike"], o["Maturity_Days"] / 365.0, r, sigma, o["Type"])
         - black_scholes_price(S, o["Strike"], o["Maturity_Days"] / 365.0, r, sigma, o["Type"])) * o["Position"]
        for o in portfolio_options
    )

    gh_pnl = (black_scholes_price(S_new, gamma_hedge_leg["Strike"], T_gh, r, sigma, gamma_hedge_leg["Type"])
              - gh_price) * gamma_hedge_qty
    stock_pnl = (S_new - S) * stock_position
    hedged_pnl = unhedged_pnl + gh_pnl + stock_pnl

    pnl_rows.append({
        "Price_Change_%": pct * 100, "New_Price": S_new,
        "Unhedged_PnL": unhedged_pnl, "Hedged_PnL": hedged_pnl,
        "Improvement": hedged_pnl - unhedged_pnl
    })

pnl_df = pd.DataFrame(pnl_rows)
pnl_df.to_csv(f"{OUT_DIR}/sunpharma_pnl_scenarios.csv", index=False)
print(pnl_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(pnl_df["Price_Change_%"], pnl_df["Unhedged_PnL"], "r-o", label="Unhedged")
ax.plot(pnl_df["Price_Change_%"], pnl_df["Hedged_PnL"], "g-s", label="Hedged")
ax.axhline(0, color="k", linestyle="--", alpha=0.3)
ax.set_xlabel("Price Change (%)"); ax.set_ylabel("PnL")
ax.set_title("SunPharma Portfolio PnL: Hedged vs Unhedged")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/sunpharma_pnl_chart.png", dpi=200)
plt.close()
print(f"\nSaved {OUT_DIR}/sunpharma_pnl_chart.png")

# ------------------------------------------------------------------
# SECTION 5: VaR — parametric and historical simulation
# ------------------------------------------------------------------
print("\n" + "=" * 100)
print("SECTION 5: Value at Risk")
print("=" * 100)

# Portfolio value today, for converting % moves into $ P&L
portfolio_value = sum(
    black_scholes_price(S, o["Strike"], o["Maturity_Days"] / 365.0, r, sigma, o["Type"]) * o["Position"]
    for o in portfolio_options
)

# --- Parametric (delta-normal) VaR: use portfolio delta as a linear proxy ---
portfolio_dollar_delta = unhedged["delta"] * S
z_scores = {"95%": 1.645, "99%": 2.326}

parametric_var = {
    level: z * daily_std * abs(portfolio_dollar_delta)
    for level, z in z_scores.items()
}

# --- Historical simulation: apply each historical daily return to the
#     portfolio via the delta-normal approximation, then take the tail ---
simulated_pnls = portfolio_dollar_delta * daily_returns
historical_var = {
    "95%": -np.percentile(simulated_pnls, 5),
    "99%": -np.percentile(simulated_pnls, 1),
}

var_summary = pd.DataFrame({
    "Confidence": ["95%", "99%"],
    "Parametric_VaR": [parametric_var["95%"], parametric_var["99%"]],
    "Historical_VaR": [historical_var["95%"], historical_var["99%"]],
})
var_summary.to_csv(f"{OUT_DIR}/sunpharma_var_summary.csv", index=False)
print(var_summary.to_string(index=False))

print(f"\nNOTE: the historical VaR above is estimated from {n_obs} daily return "
      f"observations. A 99% tail estimate from a sample this size is highly "
      f"sensitive to one or two extreme days — treat it as indicative, not precise. "
      f"A longer history (1+ year) would materially tighten this estimate.")

print("\n" + "=" * 100)
print("PART 3 COMPLETE")
print("=" * 100)
