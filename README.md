# Equity Derivatives: Pricing, Greeks, Hedging & VaR (Sun Pharma)

An options analytics pipeline built on **live NSE market data** for Sun Pharmaceutical
Industries (SUNPHARMA.NS) — Black-Scholes pricing, implied-volatility extraction, a fitted
volatility surface, a six-leg portfolio with delta/gamma hedging, and two independent VaR
estimates.

**Underlying:** Sun Pharmaceutical Industries Ltd (NSE: SUNPHARMA)
**Valuation date:** 01-Sep-2026
**Option chain:** NSE F&O, Sep-2026 and Oct-2026 expiries

## Design

Every script reads its market inputs from a single generated file, `sunpharma_output/market_inputs.json`.
Nothing downstream hardcodes a spot price, volatility, or rate — rerun Part 1 with fresh data
and the pricing grid, Greeks, hedge ratios and VaR all move with it.

All pricing math lives in exactly one module, `pricing_engine.py`. Parts 2 and 3 import from it
rather than redefining Black-Scholes locally, so there is one definition of each function with
one consistent return shape.

```
part1_data_and_market.py   ──>  sunpharma_output/market_inputs.json  ──┬──>  part2_pricing_and_greeks.py
  (the only script that                                       │       (IV, Greeks, vol surface)
   touches Yahoo Finance)                                      │
                                                              └──>  part3_portfolio_and_risk.py
pricing_engine.py  ──── imported by Parts 2 and 3 ────                (hedging, PnL, VaR)
```

## Files

| File | What it is |
|---|---|
| `pricing_engine.py` | Shared module: Black-Scholes pricing, the five Greeks, implied volatility via Brent's method, day-count conversion, moneyness classification. |
| `part1_data_and_market.py` | Downloads 3 months of price history, computes return statistics (mean, std, annualized vol, skew, kurtosis), fetches the option chain, and writes `sunpharma_output/market_inputs.json` — the single source of truth for everything downstream. |
| `part2_pricing_and_greeks.py` | Black-Scholes pricing grid across strikes × maturities; implied volatility extracted from the real chain; Greeks computed under both historical and implied vol; 3-D volatility surface. |
| `part3_portfolio_and_risk.py` | Builds a six-leg option portfolio, delta-hedges then gamma-hedges then re-nulls delta, simulates PnL across spot scenarios hedged vs unhedged, and computes parametric and historical VaR. |
| `parse_nse_option_chain.py` | Reshapes NSE's two-header-row option-chain export (calls left, strike centre, puts right) into the flat `Type, Strike, LTP, Volume, Expiry` schema the rest of the project expects. |
| `SunPharma_Derivatives_Project.xlsx` | 11-sheet workbook presenting the same analysis: Dashboard, Price History, Option Chain, Implied Vols, Volatility Surface, Pricing Grid, Greeks, Portfolio Composition, Hedge Summary, PnL Scenarios, VaR Summary. |
| `sunpharma_output/` | Generated artefacts — CSVs, `market_inputs.json`, the volatility-surface and PnL charts. |

## Running it

```bash
pip install numpy pandas scipy matplotlib yfinance openpyxl

python3 part1_data_and_market.py      # writes sunpharma_output/market_inputs.json
python3 part2_pricing_and_greeks.py   # pricing grid, IV, Greeks, vol surface
python3 part3_portfolio_and_risk.py   # hedging, PnL scenarios, VaR
```

Yahoo Finance does not serve F&O chain data for most NSE tickers, so Part 1 will report an
empty chain for SUNPHARMA.NS. Download the chain from the NSE option-chain page and run
`parse_nse_option_chain.py` over the exports — point `NSE_FILES` at your downloaded CSVs first.

## Market inputs

| Input | Value | Note |
|---|---|---|
| Spot | ₹1,895.17 | Implied by put-call parity across six near-ATM strikes on the Sep-2026 chain |
| Historical volatility | 14.39% annualized | 65 trading days of log returns |
| Risk-free rate | 6.50% | Short-term G-Sec proxy |
| Return skew / kurtosis | 0.57 / 5.36 | Fat-tailed and right-skewed over the window |

Yahoo's last close (₹1,984.80) was stale relative to the option-chain snapshot. Using the
parity-implied spot instead keeps pricing, Greeks and VaR internally consistent with the chain
they are computed from — the workbook's `Price History` sheet still carries the Yahoo series.

## Results

**Hedging.** The six-leg portfolio carried +11.91 delta and +0.0356 gamma before hedging. A
stock-and-option hedge takes both to zero:

| Greek | Before hedge | After hedge |
|---|---|---|
| Delta | 11.91 | 0.00 |
| Gamma | 0.0356 | 0.00 |
| Vega | 20.26 | −25.10 |
| Theta | −7.58 | −1.57 |

Delta and gamma are neutralized, but the hedge flips vega negative — the position is now short
volatility. That is the trade-off the exercise is meant to expose: neutralizing the directional
Greeks does not make a book riskless, it relocates the risk into vega.

**PnL under the hedge.** Across a ±2% spot move, unhedged PnL swings from −427 to +477. Hedged
PnL stays inside ±1. The hedge removes the downside and the upside in equal measure.

**Value at Risk**, one-day, on the unhedged book:

| Confidence | Parametric | Historical |
|---|---|---|
| 95% | 336.66 | 256.35 |
| 99% | 476.02 | 401.97 |

Parametric VaR runs above historical at both levels because the normal assumption behind it
does not fit a return series with kurtosis of 5.36. The 99% historical figure rests on 65
observations — roughly the worst one of them — so it should be read as indicative rather than
precise.

## Data source

Price history via Yahoo Finance (`yfinance`). Option chain from the NSE option-chain export,
Sep-2026 and Oct-2026 expiries. Risk-free rate is a short-term Indian G-Sec proxy.

## Caveats

- Black-Scholes assumes constant volatility and no early exercise. NSE stock options are
  American-style, so the prices here are European approximations.
- No dividend adjustment is applied to the underlying.
- The vol surface is interpolated (`scipy.griddata`) across a sparse set of liquid strikes, not
  fitted with a parametric model such as SVI.
- VaR is one-day and assumes positions are held static.
