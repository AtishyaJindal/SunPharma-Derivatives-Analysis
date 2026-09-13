
"""
NSE OPTION CHAIN PARSER — SunPharma Derivatives Project
==========================================================
NSE's option-chain CSV export is a wide, two-header-row format with
CALLS on the left, STRIKE in the middle, and PUTS on the right
(duplicate column names for BID/ASK/LTP/IV/VOLUME/OI on each side).

This script reshapes that into the flat schema every other file in this
project expects:  Type, Strike, LTP, Volume, Expiry

Run this once per NSE export, then feed the combined CSV into
part1_data_and_market.py's option-chain step (or drop it straight into
sunpharma_output/sunpharma_option_chain.csv to skip re-running Part 1's
yfinance section).

USAGE
-----
    python3 parse_nse_option_chain.py

Edit NSE_FILES below to point at your downloaded exports. The expiry
date is taken from each filename (NSE names them
'option-chain-ED-<SYMBOL>-<DD-Mon-YYYY>.csv').
"""

import re
import pandas as pd

NSE_FILES = [
    "./option-chain-ED-SUNPHARMA-29-Sep-2026.csv",
    "./option-chain-ED-SUNPHARMA-27-Oct-2026.csv",
]
OUT_PATH = "./sunpharma_output/sunpharma_option_chain.csv"


def parse_expiry_from_filename(path):
    """Extract '27-Oct-2026' style date from the NSE export filename and
    convert to ISO 'YYYY-MM-DD', matching the schema pricing_engine.py expects."""
    m = re.search(r"(\d{1,2}-[A-Za-z]{3}-\d{4})", path)
    if not m:
        raise ValueError(f"Could not find a date in filename: {path}")
    return pd.to_datetime(m.group(1), format="%d-%b-%Y").strftime("%Y-%m-%d")


def to_float(x):
    """NSE exports numbers as strings with thousands-separators and '-'
    for no-data cells. Convert cleanly, returning None for '-'."""
    if pd.isna(x):
        return None
    x = str(x).strip()
    if x == "-" or x == "":
        return None
    return float(x.replace(",", ""))


def parse_one_file(path):
    expiry = parse_expiry_from_filename(path)
    raw = pd.read_csv(path, header=1)

    rows = []
    for _, r in raw.iterrows():
        strike = to_float(r["STRIKE"])
        if strike is None:
            continue

        call_ltp = to_float(r["LTP"])
        call_vol = to_float(r["VOLUME"]) or 0
        if call_ltp is not None:
            rows.append({"Type": "Call", "Strike": strike, "LTP": call_ltp,
                         "Volume": int(call_vol), "Expiry": expiry})

        put_ltp = to_float(r["LTP.1"])
        put_vol = to_float(r["VOLUME.1"]) or 0
        if put_ltp is not None:
            rows.append({"Type": "Put", "Strike": strike, "LTP": put_ltp,
                         "Volume": int(put_vol), "Expiry": expiry})

    print(f"{path}: parsed {len(rows)} live-quoted contracts for expiry {expiry}")
    return rows


if __name__ == "__main__":
    all_rows = []
    for path in NSE_FILES:
        all_rows.extend(parse_one_file(path))

    df = pd.DataFrame(all_rows).sort_values(["Expiry", "Type", "Strike"]).reset_index(drop=True)

    import os
    os.makedirs("./sunpharma_output", exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\nTotal contracts written: {len(df)}")
    print(df.to_string(index=False))
    print(f"\nSaved to {OUT_PATH}")
