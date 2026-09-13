"""
PRICING ENGINE — SunPharma Derivatives Project
================================================
Single shared module for Black-Scholes pricing, Greeks, implied volatility,
day-count conversion, and moneyness classification.

Every other script in this project (part2_pricing_and_greeks.py,
part3_portfolio_and_risk.py) imports from HERE instead of redefining these
functions locally. This is the fix for the original project's biggest
structural issue: `black_scholes_price()` was copy-pasted into six files,
and the Greeks function returned a different shape (dict vs. 5-tuple vs.
3-tuple) depending on which file you looked at. Here there is exactly one
definition of each, with one consistent return shape, used everywhere.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


# ------------------------------------------------------------------
# Pricing
# ------------------------------------------------------------------
def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """
    Black-Scholes-Merton price for a European call or put.

    Parameters
    ----------
    S : float   Spot price
    K : float   Strike price
    T : float   Time to maturity, in YEARS (use year_fraction() to compute this)
    r : float   Risk-free rate (annualized, continuously compounded)
    sigma : float   Volatility (annualized)
    option_type : {'call', 'put'}

    Returns
    -------
    float — option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


# ------------------------------------------------------------------
# Greeks — ALWAYS returns all five, for both calls and puts
# ------------------------------------------------------------------
def greeks(S, K, T, r, sigma, option_type="call"):
    """
    Black-Scholes Greeks. Returns a dict with ALL FIVE keys every time,
    regardless of option_type — this is what keeps every downstream script
    consistent (the original project's PartC returned a dict, PartD a
    5-tuple, PartE a 3-tuple with no theta/rho; that inconsistency is why
    fixes never propagated cleanly).

    Returns
    -------
    dict with keys: 'delta', 'gamma', 'vega', 'theta', 'rho'
        - delta: dimensionless, per 1-unit move in S
        - gamma: dimensionless, per 1-unit move in S
        - vega:  price change per 1% (0.01) move in sigma
        - theta: price change per calendar day (already divided by 365)
        - rho:   price change per 1% (0.01) move in r
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    is_call = option_type.lower() == "call"

    if is_call:
        delta = norm.cdf(d1)
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    elif option_type.lower() == "put":
        delta = norm.cdf(d1) - 1
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


# ------------------------------------------------------------------
# Implied volatility
# ------------------------------------------------------------------
def implied_volatility(market_price, S, K, T, r, option_type="call"):
    """
    Solve for the Black-Scholes implied volatility using Brent's method.
    Returns np.nan (not a silent exception) if no solution is found in
    [0.01, 3.00] — e.g. for a price outside arbitrage bounds.
    """
    def objective(sigma):
        return black_scholes_price(S, K, T, r, sigma, option_type) - market_price

    try:
        return brentq(objective, 0.01, 3.0, maxiter=100)
    except ValueError:
        return np.nan


# ------------------------------------------------------------------
# Day count — single source of truth for every T in the project
# ------------------------------------------------------------------
def year_fraction(valuation_date, expiry_date):
    """
    Actual-days-over-365 year fraction between two datetime.date /
    datetime.datetime objects. Every T in this project traces back to
    this function — never a hardcoded literal or a comment-guessed
    constant (the original project's Q7/Q8 hardcoded T=38/365 with a
    comment claiming ~32 days; Q9 computed it correctly from real dates
    and got 36. This function removes that inconsistency entirely.)
    """
    return (expiry_date - valuation_date).days / 365.0


# ------------------------------------------------------------------
# Moneyness — correct logic for calls AND puts
# ------------------------------------------------------------------
def classify_moneyness(strike, spot, option_type, band=0.02):
    """
    Classify a contract as 'ITM', 'ATM', or 'OTM'.

    CALL: ITM if strike < spot, OTM if strike > spot.
    PUT:  ITM if strike > spot, OTM if strike < spot.  <-- opposite of call

    The original project applied the call rule to puts too, silently
    swapping every put's ITM/OTM tag. `band` sets the ATM tolerance as a
    fraction of spot (default 2%).
    """
    moneyness = strike / spot
    lower, upper = 1 - band, 1 + band

    if lower <= moneyness <= upper:
        return "ATM"

    if option_type.lower() == "call":
        return "ITM" if moneyness < 1 else "OTM"
    elif option_type.lower() == "put":
        return "ITM" if moneyness > 1 else "OTM"
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
