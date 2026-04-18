"""
mt5_connector.py
────────────────
Handles all communication with the MetaTrader 5 terminal.
- Connects / disconnects
- Fetches live OHLCV candle data
- Retrieves account info
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import time


# ── Symbols the bot will trade ──────────────────────────────────────────────
SYMBOLS = [
    # Forex
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF", "EURJPY", "GBPJPY", "EURGBP",
    # Indexes
    "US30", "NAS100", "SPX500", "GER40", "UK100",
    # Stocks
    "AAPL", "TSLA", "NVDA", "MSFT", "AMZN",
]

# ── Candle timeframe ─────────────────────────────────────────────────────────
TIMEFRAME = mt5.TIMEFRAME_M15   # 15-minute candles (good for scalping)
CANDLES   = 200                  # how many candles to fetch per symbol


def connect() -> bool:
    """Start MT5 and confirm login. Returns True if successful."""
    if not mt5.initialize():
        print(f"[ERROR] MT5 init failed: {mt5.last_error()}")
        return False

    info = mt5.account_info()
    if info is None:
        print(f"[ERROR] Cannot read account: {mt5.last_error()}")
        return False

    print("=" * 50)
    print(f"  Connected to MT5")
    print(f"  Account : {info.login}")
    print(f"  Broker  : {info.company}")
    print(f"  Balance : ${info.balance:,.2f}")
    print(f"  Equity  : ${info.equity:,.2f}")
    print("=" * 50)
    return True


def disconnect():
    """Cleanly shut down the MT5 connection."""
    mt5.shutdown()
    print("[INFO] MT5 disconnected.")


def get_account_info() -> dict:
    """Return key account figures as a plain dict."""
    info = mt5.account_info()
    if info is None:
        return {}
    return {
        "balance"   : info.balance,
        "equity"    : info.equity,
        "margin"    : info.margin,
        "free_margin": info.margin_free,
        "profit"    : info.profit,
        "leverage"  : info.leverage,
    }


def fetch_candles(symbol: str, timeframe=TIMEFRAME, n=CANDLES) -> pd.DataFrame:
    """
    Pull the last *n* closed candles for *symbol*.
    Returns a DataFrame with columns: time, open, high, low, close, volume.
    Returns an empty DataFrame on failure.
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if rates is None or len(rates) == 0:
        print(f"[WARN] No data for {symbol}: {mt5.last_error()}")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df = df[["open", "high", "low", "close", "tick_volume"]].rename(
        columns={"tick_volume": "volume"}
    )
    return df


def fetch_all_symbols() -> dict[str, pd.DataFrame]:
    """Fetch candles for every symbol in SYMBOLS. Returns {symbol: df}."""
    data = {}
    for sym in SYMBOLS:
        df = fetch_candles(sym)
        if not df.empty:
            data[sym] = df
    return data


def get_open_positions() -> list[dict]:
    """Return all currently open trades as a list of dicts."""
    positions = mt5.positions_get()
    if positions is None:
        return []
    result = []
    for p in positions:
        result.append({
            "ticket"    : p.ticket,
            "symbol"    : p.symbol,
            "type"      : "BUY" if p.type == 0 else "SELL",
            "volume"    : p.volume,
            "open_price": p.price_open,
            "sl"        : p.sl,
            "tp"        : p.tp,
            "profit"    : p.profit,
        })
    return result


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if connect():
        print("\nFetching EURUSD candles …")
        df = fetch_candles("EURUSD")
        print(df.tail(5))

        print("\nAccount info:")
        for k, v in get_account_info().items():
            print(f"  {k}: {v}")

        disconnect()
