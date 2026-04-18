"""
risk_manager.py  ── v3 (FIXED)
────────────────────────────────
Calculates position sizing, stop loss, and take profit.

v3 fixes
  ✔ REWARD_RATIO restored to 2.0  (1:1 R:R from v2.1 is a losing strategy long-term)
  ✔ ATR_MULTIPLIER stays at 1.5   (wider SL prevents whipsaw)
  ✔ MIN_SL_PIPS = 5               (spread protection)
  ✔ MAX_OPEN_TRADES = 8           (protects $1,000 account margin)
  ✔ Margin check in can_open_trade
"""

import MetaTrader5 as mt5
import numpy as np
import pandas as pd


RISK_PCT        = 0.005   # 0.5% risk per trade
REWARD_RATIO    = 2.0     # TP = 2× SL → 1:2 R:R  (was 1.0 in v2.1 — restored)
ATR_PERIOD      = 14
ATR_MULTIPLIER  = 1.5     # wider stop, prevents whipsaw
MAX_OPEN_TRADES = 8
MIN_LOT         = 0.01
MAX_LOT         = 0.5
MIN_SL_PIPS     = 5


def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    high  = df["high"]
    low   = df["low"]
    close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low  - close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    return float(atr.iloc[-1])


def get_pip_value(symbol: str, lot: float = 1.0) -> float:
    info = mt5.symbol_info(symbol)
    if info is None:
        return 10.0
    pip_size = info.point * 10 if "JPY" not in symbol else info.point * 1000
    tick_val = info.trade_tick_value
    tick_sz  = info.trade_tick_size
    if tick_sz == 0:
        return 10.0
    return round((pip_size / tick_sz) * tick_val * lot, 4)


def calculate_lot_size(symbol: str, sl_pips: float, balance: float) -> float:
    if sl_pips <= 0:
        return MIN_LOT
    risk_amount   = balance * RISK_PCT
    pip_val_1_lot = get_pip_value(symbol, 1.0)
    if pip_val_1_lot <= 0:
        return MIN_LOT
    raw_lot = risk_amount / (sl_pips * pip_val_1_lot)
    lot     = max(MIN_LOT, min(MAX_LOT, raw_lot))
    info = mt5.symbol_info(symbol)
    step = info.volume_step if info else 0.01
    return round(round(lot / step) * step, 2)


def calculate_sl_tp(
    symbol     : str,
    direction  : str,
    df         : pd.DataFrame,
    entry_price: float = None,
) -> dict:
    atr         = compute_atr(df)
    sl_distance = atr * ATR_MULTIPLIER

    if entry_price is None:
        tick = mt5.symbol_info_tick(symbol)
        entry_price = tick.ask if direction == "BUY" else tick.bid

    info    = mt5.symbol_info(symbol)
    digits  = info.digits if info else 5
    pip_sz  = 0.0001 if (info and "JPY" not in symbol) else 0.01

    sl_pips = sl_distance / pip_sz
    if sl_pips < MIN_SL_PIPS:
        sl_pips     = MIN_SL_PIPS
        sl_distance = MIN_SL_PIPS * pip_sz

    if direction == "BUY":
        sl = round(entry_price - sl_distance, digits)
        tp = round(entry_price + sl_distance * REWARD_RATIO, digits)
    else:
        sl = round(entry_price + sl_distance, digits)
        tp = round(entry_price - sl_distance * REWARD_RATIO, digits)

    acct    = mt5.account_info()
    balance = acct.balance if acct else 1000.0
    lot     = calculate_lot_size(symbol, sl_pips, balance)

    return {"sl": sl, "tp": tp, "sl_pips": round(sl_pips, 1), "lot": lot, "entry": entry_price}


def can_open_trade(symbol: str) -> tuple[bool, str]:
    positions = mt5.positions_get()
    n_open = len(positions) if positions else 0

    if n_open >= MAX_OPEN_TRADES:
        return False, f"Max open trades ({n_open}/{MAX_OPEN_TRADES})"

    if positions and symbol in [p.symbol for p in positions]:
        return False, f"Already in {symbol}"

    info = mt5.symbol_info(symbol)
    if info is None or not info.visible:
        return False, f"{symbol} not available"

    acct = mt5.account_info()
    if acct and acct.margin_free < acct.balance * 0.10:
        return False, f"Low free margin ({acct.margin_free:.2f})"

    return True, ""
