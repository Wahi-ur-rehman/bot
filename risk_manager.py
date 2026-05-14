"""
risk_manager.py  ── v4 (Production)
──────────────────────────────────────
v4 upgrades over v3:
  ✔ PerformanceTracker  — rolling win rate, avg win/loss in R-multiples
  ✔ Kelly Criterion position sizing (25% fractional) with fallback
  ✔ Adaptive SL/TP multipliers per market regime
      TRENDING  → SL×1.5, TP×4.0   (let winners run)
      RANGING   → SL×1.2, TP×1.8   (quick scalps)
      VOLATILE  → SL×2.5, TP×3.5   (wider stops)
      NEUTRAL   → SL×1.5, TP×2.0   (balanced)
  ✔ update_trailing_stop() — moves SL once price advances 1 ATR
  ✔ MIN_KELLY_TRADES = 10   (use fixed sizing until enough history)
"""

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from collections import deque


# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
RISK_PCT         = 0.005    # 0.5% risk per trade (fixed fallback)
REWARD_RATIO     = 2.0      # default TP = 2× SL  (overridden by regime)
ATR_PERIOD       = 14
MAX_OPEN_TRADES  = 6        # reduced from 8 — tighter exposure at M5
MIN_LOT          = 0.01
MAX_LOT          = 0.5
MIN_SL_PIPS      = 5
KELLY_FRACTION   = 0.25     # use 25% of full Kelly — much safer
MIN_KELLY_TRADES = 10       # minimum trades before Kelly activates

# SL / TP multipliers per regime
REGIME_PARAMS = {
    "TRENDING":  {"sl_mult": 1.5, "tp_mult": 4.0},
    "RANGING":   {"sl_mult": 1.2, "tp_mult": 1.8},
    "VOLATILE":  {"sl_mult": 2.5, "tp_mult": 3.5},
    "NEUTRAL":   {"sl_mult": 1.5, "tp_mult": 2.0},
}


# ══════════════════════════════════════════════════════════════════════
# PERFORMANCE TRACKER
# ══════════════════════════════════════════════════════════════════════
class PerformanceTracker:
    """
    Tracks recent trade outcomes to compute win rate and
    avg win/loss in R-multiples for Kelly position sizing.

    Usage:
        tracker = PerformanceTracker()
        tracker.record("TP_HIT", pnl=25.0, sl_distance=12.0)
        lot = tracker.kelly_lot(symbol, balance)
    """

    def __init__(self, window: int = 50):
        self._outcomes: deque = deque(maxlen=window)  # True=win, False=loss
        self._r_wins:   deque = deque(maxlen=window)  # R-multiple on wins
        self._r_losses: deque = deque(maxlen=window)  # R-multiple on losses

    def record(self, result: str, pnl: float, sl_distance_usd: float = None):
        """
        Call this on every closed trade.
        result: "TP_HIT" | "SL_HIT" | "MANUAL_CLOSE"
        sl_distance_usd: approximate dollar value of 1 SL distance (for R calc)
        """
        is_win = result == "TP_HIT" or pnl > 0

        self._outcomes.append(1 if is_win else 0)

        if sl_distance_usd and sl_distance_usd > 0:
            r = abs(pnl) / sl_distance_usd
            if is_win:
                self._r_wins.append(r)
            else:
                self._r_losses.append(r)

    @property
    def trade_count(self) -> int:
        return len(self._outcomes)

    @property
    def win_rate(self) -> float:
        if not self._outcomes:
            return 0.5
        return sum(self._outcomes) / len(self._outcomes)

    @property
    def avg_win_r(self) -> float:
        return float(np.mean(self._r_wins)) if self._r_wins else 2.0

    @property
    def avg_loss_r(self) -> float:
        return float(np.mean(self._r_losses)) if self._r_losses else 1.0

    def kelly_fraction_raw(self) -> float:
        """
        Full Kelly: f = (W / L) - ((1-W) / W_avg)
        where W = win_rate, L = avg_loss_r, W_avg = avg_win_r
        """
        w = self.win_rate
        b = self.avg_win_r    # avg win in R
        q = 1 - w
        a = self.avg_loss_r   # avg loss in R
        if a == 0:
            return 0.0
        return (w * b - q * a) / (a * b) if (a * b) > 0 else 0.0

    def print_stats(self):
        print(f"  [TRACKER] Trades={self.trade_count}  "
              f"Win%={self.win_rate:.1%}  "
              f"AvgWin={self.avg_win_r:.2f}R  "
              f"AvgLoss={self.avg_loss_r:.2f}R  "
              f"Kelly={self.kelly_fraction_raw():.3f}")


# Singleton — shared across bot.py and risk_manager
_perf_tracker = PerformanceTracker()


def get_performance_tracker() -> PerformanceTracker:
    return _perf_tracker


# ══════════════════════════════════════════════════════════════════════
# ATR CALCULATION
# ══════════════════════════════════════════════════════════════════════
def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    # Use precomputed column if available (faster)
    if "atr" in df.columns:
        val = df["atr"].iloc[-1]
        if pd.notna(val) and val > 0:
            return float(val)

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


# ══════════════════════════════════════════════════════════════════════
# PIP VALUE
# ══════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════
# LOT SIZE
# ══════════════════════════════════════════════════════════════════════
def calculate_lot_size(
    symbol: str,
    sl_pips: float,
    balance: float,
    use_kelly: bool = True,
) -> float:
    """
    Calculate position size.
    - If Kelly is active (enough trades): uses fractional Kelly %.
    - Otherwise: falls back to fixed RISK_PCT of balance.
    """
    if sl_pips <= 0:
        return MIN_LOT

    info    = mt5.symbol_info(symbol)
    step    = info.volume_step if info else 0.01

    pip_val_1_lot = get_pip_value(symbol, 1.0)
    if pip_val_1_lot <= 0:
        return MIN_LOT

    # ─── Kelly sizing ────────────────────────────────────────────────
    tracker = get_performance_tracker()

    if use_kelly and tracker.trade_count >= MIN_KELLY_TRADES:
        raw_kelly   = tracker.kelly_fraction_raw()
        kelly_pct   = max(0.001, min(0.05, raw_kelly * KELLY_FRACTION))
        risk_amount = balance * kelly_pct
    else:
        risk_amount = balance * RISK_PCT

    # ─── Convert risk amount to lots ─────────────────────────────────
    raw_lot = risk_amount / (sl_pips * pip_val_1_lot)
    lot     = max(MIN_LOT, min(MAX_LOT, raw_lot))
    return round(round(lot / step) * step, 2)


# ══════════════════════════════════════════════════════════════════════
# SL / TP CALCULATION
# ══════════════════════════════════════════════════════════════════════
def calculate_sl_tp(
    symbol      : str,
    direction   : str,
    df          : pd.DataFrame,
    entry_price : float = None,
    regime      : str   = "NEUTRAL",
) -> dict:
    """
    Compute SL/TP/lot with regime-adaptive multipliers.

    Args:
        symbol: MT5 symbol name
        direction: "BUY" or "SELL"
        df: OHLCV DataFrame (used for ATR)
        entry_price: override entry (defaults to current ask/bid)
        regime: "TRENDING"|"RANGING"|"VOLATILE"|"NEUTRAL"

    Returns:
        dict: sl, tp, sl_pips, lot, entry, sl_mult, tp_mult, regime
    """
    atr = compute_atr(df)

    params    = REGIME_PARAMS.get(regime, REGIME_PARAMS["NEUTRAL"])
    sl_mult   = params["sl_mult"]
    tp_mult   = params["tp_mult"]

    sl_distance = atr * sl_mult

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
        tp = round(entry_price + sl_distance * tp_mult, digits)
    else:
        sl = round(entry_price + sl_distance, digits)
        tp = round(entry_price - sl_distance * tp_mult, digits)

    acct    = mt5.account_info()
    balance = acct.balance if acct else 1000.0
    lot     = calculate_lot_size(symbol, sl_pips, balance)

    return {
        "sl":      sl,
        "tp":      tp,
        "sl_pips": round(sl_pips, 1),
        "lot":     lot,
        "entry":   entry_price,
        "sl_mult": sl_mult,
        "tp_mult": tp_mult,
        "regime":  regime,
        "atr":     round(atr, 6),
    }


# ══════════════════════════════════════════════════════════════════════
# TRAILING STOP
# ══════════════════════════════════════════════════════════════════════
def update_trailing_stop(
    ticket         : int,
    direction      : str,
    current_price  : float,
    atr            : float,
    trail_mult     : float = 1.0,
) -> bool:
    """
    Move SL to trail price by trail_mult × ATR.
    Only moves SL in the profitable direction (never widens it).

    Returns True if SL was modified, False otherwise.
    """
    position = mt5.positions_get(ticket=ticket)
    if not position:
        return False

    pos         = position[0]
    current_sl  = pos.sl
    symbol      = pos.symbol
    info        = mt5.symbol_info(symbol)
    digits      = info.digits if info else 5

    trail_distance = atr * trail_mult

    if direction == "BUY":
        new_sl = round(current_price - trail_distance, digits)
        if new_sl <= current_sl:
            return False   # would widen the stop — don't do it
    else:
        new_sl = round(current_price + trail_distance, digits)
        if new_sl >= current_sl:
            return False

    request = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "sl":       new_sl,
        "tp":       pos.tp,
    }
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        return True
    return False


# ══════════════════════════════════════════════════════════════════════
# TRADE GATE
# ══════════════════════════════════════════════════════════════════════
def can_open_trade(symbol: str) -> tuple[bool, str]:
    positions = mt5.positions_get()
    n_open    = len(positions) if positions else 0

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
