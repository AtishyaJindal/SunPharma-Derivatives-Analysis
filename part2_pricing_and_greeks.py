"""
PART 2 — PRICING, IMPLIED VOLATILITY, GREEKS, VOL SURFACE
============================================================
Replaces the original project's Part B + Q7 + Q8 + Q9 with one script.

Reads `market_inputs.json` and `sunpharma_option_chain.csv` produced by
Part 1 — no hardcoded spot/vol/rate/strikes anywhere in this file.
Imports all pricing math from `pricing_engine.py` — no redefinitions.

SECTIONS
--------
1. Load market inputs & option chain
2. BS pricing grid across strikes x maturities (was Part B / Q4-6)
3. Implied volatility extraction from the real option chain (was Q7 Step 1)
4. Historical-vol vs implied-vol Greeks — ONE combined CSV with a
   Volatility_Type column (was Q7+Q8, which wrongly split this into two
   scripts writing to the same filename)
5. Volatility surface across strikes and maturities (was Q9)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d projection)

from pricing_engine import (
    black_scholes_price, greeks, implied_volatility,
    year_fraction, classify_moneyness
)

IN_DIR = "./sunpharma_output"
OUT_DIR = "./sunpharma_output"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# SECTION 1: Load market inputs & option chain
# ------------------------------------------------------------------
print("=" * 100)
print("SECTION 1: Loading market inputs")
print("=" * 100)

with open(f"{IN_DIR}/market_inputs.json") as f:
    mkt = json.load(f)

S = mkt["spot"]
r = mkt["risk_free_rate"]
sigma_hist = mkt["historical_volatility"]
valuation_date = datetime.strptime(mkt["valuation_date"], "%Y-%m-%d")
strikes = mkt["strikes_for_analysis"]
maturities_days = mkt["maturities_days"]

print(f"Spot: {S} | Historical Vol: {sigma_hist*100:.2f}% | r: {r*100:.2f}% "
      f"| Valuation Date: {mkt['valuation_date']}")
print(f"Strikes: {strikes}")
print(f"Maturities (days): {maturities_days}")

option_chain = pd.read_csv(f"{IN_DIR}/sunpharma_option_chain.csv")
if option_chain.empty:
    print("\nWARNING: option chain is empty (see part1's Section 4 note). "
          "Sections 3, 4, and 5 below need real contracts to run — fill in "
          "MANUAL_OPTION_CHAIN in part1_data_and_market.py and rerun Part 1 first.")

# ------------------------------------------------------------------
# SECTION 2: BS pricing grid (strikes x maturities)
# ------------------------------------------------------------------
print("\n" + "=" * 100)
print("SECTION 2: Black-Scholes pricing grid")
print("=" * 100)

pricing_rows = []
for K in strikes:
    for days in maturities_days:
        T = days / 365.0
        for opt_type in ("call", "put"):
            price = black_scholes_price(S, K, T, r, sigma_hist, opt_type)
            intrinsic = max(S - K, 0) if opt_type == "call" else max(K - S, 0)
            pricing_rows.append({
                "Strike": K, "Maturity_Days": days, "Type": opt_type.capitalize(),
                "Price": price, "Intrinsic_Value": intrinsic,
                "Time_Value": price - intrinsic,
                "Moneyness": classify_moneyness(K, S, opt_type)  # correct for both calls & puts
            })

pricing_df = pd.DataFrame(pricing_rows)
pricing_df.to_csv(f"{OUT_DIR}/sunpharma_pricing_grid.csv", index=False)
print(pricing_df.to_string(index=False))
print(f"\nSaved to {OUT_DIR}/sunpharma_pricing_grid.csv")

# Put-Call parity check
print("\nPut-Call Parity check (Call - Put = S - K*e^-rT):")
for K in strikes:
    for days in maturities_days:
        T = days / 365.0
        call_p = pricing_df[(pricing_df.Strike == K) & (pricing_df.Maturity_Days == days)
                             & (pricing_df.Type == "Call")]["Price"].iloc[0]
        put_p = pricing_df[(pricing_df.Strike == K) & (pricing_df.Maturity_Days == days)
                            & (pricing_df.Type == "Put")]["Price"].iloc[0]
        lhs, rhs = call_p - put_p, S - K * np.exp(-r * T)
        print(f"  K={K:9.2f} T={days:3d}d  C-P={lhs:9.4f}  S-Ke^-rT={rhs:9.4f}  "
              f"diff={abs(lhs-rhs):.6f}")

# ------------------------------------------------------------------
# SECTION 3: Implied volatility from the real option chain
# ------------------------------------------------------------------
avg_iv = None
if not option_chain.empty:
    print("\n" + "=" * 100)
    print("SECTION 3: Implied volatility extraction")
    print("=" * 100)

    iv_rows = []
    for _, row in option_chain.iterrows():
        expiry = datetime.strptime(str(row["Expiry"]), "%Y-%m-%d")
        T = year_fraction(valuation_date, expiry)   # real dates, never a hardcoded day count
        if T <= 0:
            continue
        iv = implied_volatility(row["LTP"], S, row["Strike"], T, r, row["Type"].lower())
        if not np.isnan(iv):
            iv_rows.append({**row.to_dict(), "T": T, "Days_to_Expiry": round(T * 365),
                             "Implied_Vol": iv})
            print(f"{row['Expiry']} | {row['Type']:4s} | Strike {row['Strike']:8.1f} "
                  f"| LTP {row['LTP']:7.2f} | IV {iv*100:6.2f}%")

    iv_df = pd.DataFrame(iv_rows)
    iv_df.to_csv(f"{OUT_DIR}/sunpharma_implied_vols.csv", index=False)

    if not iv_df.empty:
        avg_iv = float(iv_df["Implied_Vol"].mean())
        print(f"\nAverage Implied Volatility: {avg_iv*100:.2f}%")
        print(f"Historical Volatility:      {sigma_hist*100:.2f}%")
        print(f"Difference:                 {(avg_iv - sigma_hist)*100:.2f} pp")

# ------------------------------------------------------------------
# SECTION 4: Historical-vol vs implied-vol Greeks — ONE combined CSV
# ------------------------------------------------------------------
if avg_iv is not None:
    print("\n" + "=" * 100)
    print("SECTION 4: Greeks — historical vol vs implied vol (single combined file)")
    print("=" * 100)

    greek_rows = []
    for vol_type, sigma_used in (("Historical", sigma_hist), ("Implied", avg_iv)):
        for K in strikes:
            for days in maturities_days:
                T = days / 365.0
                for opt_type in ("call", "put"):
                    g = greeks(S, K, T, r, sigma_used, opt_type)
                    greek_rows.append({
                        "Volatility_Type": vol_type,   # <-- the fix: one file, tagged column
                        "Volatility_Used": sigma_used,
                        "Strike": K, "Maturity_Days": days, "Type": opt_type.capitalize(),
                        **g
                    })

    greeks_df = pd.DataFrame(greek_rows)
    greeks_df.to_csv(f"{OUT_DIR}/sunpharma_greeks_hist_vs_iv.csv", index=False)
    print(f"Saved {len(greeks_df)} rows to {OUT_DIR}/sunpharma_greeks_hist_vs_iv.csv")
    print("(filter this file by Volatility_Type to get the 'historical' view or "
          "the 'implied' view — no duplicate scripts, no overwritten output)")

    # Quick comparison summary
    pivot = greeks_df.pivot_table(index=["Strike", "Maturity_Days", "Type"],
                                   columns="Volatility_Type",
                                   values=["delta", "gamma", "vega", "theta", "rho"])
    print("\nSample comparison (first 5 rows):")
    print(pivot.head().to_string())

# ------------------------------------------------------------------
# SECTION 5: Volatility surface
# ------------------------------------------------------------------
if avg_iv is not None and len(iv_df) >= 4:
    print("\n" + "=" * 100)
    print("SECTION 5: Volatility surface")
    print("=" * 100)

    strikes_arr = iv_df["Strike"].values
    days_arr = iv_df["Days_to_Expiry"].values
    ivs_arr = iv_df["Implied_Vol"].values * 100

    strike_grid = np.linspace(strikes_arr.min(), strikes_arr.max(), 100)
    days_grid = np.linspace(days_arr.min(), days_arr.max(), 100)
    strike_mesh, days_mesh = np.meshgrid(strike_grid, days_grid)
    iv_mesh = griddata((strikes_arr, days_arr), ivs_arr, (strike_mesh, days_mesh), method="cubic")

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(strike_mesh, days_mesh, iv_mesh, cmap="viridis",
                            alpha=0.9, edgecolor="none")
    ax.scatter(strikes_arr, days_arr, ivs_arr, c="red", s=60, label="Market Data")
    ax.set_xlabel("Strike"); ax.set_ylabel("Days to Expiry"); ax.set_zlabel("IV (%)")
    ax.set_title("SunPharma Volatility Surface")
    fig.colorbar(surf, shrink=0.5, aspect=10)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/sunpharma_vol_surface.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT_DIR}/sunpharma_vol_surface.png")
else:
    print("\nSkipping volatility surface: need at least 4 valid IV points across "
          "multiple expiries. Add more option chain data (ideally 2+ expiries) to enable this.")

print("\n" + "=" * 100)
print("PART 2 COMPLETE")
print("=" * 100)
