"""
logger.py
─────────
Writes all bot activity to:
  - Console (colour-coded)
  - trade_log.csv  (full trade history)
  - signal_log.csv (all signals generated, including HOLDs)
"""

import csv
import os
from datetime import datetime


LOG_DIR        = os.path.dirname(os.path.abspath(__file__))
TRADE_LOG_FILE = os.path.join(LOG_DIR, "trade_log.csv")
SIGNAL_LOG_FILE= os.path.join(LOG_DIR, "signal_log.csv")

TRADE_HEADERS  = ["timestamp","symbol","direction","lot","entry","sl","tp",
                   "ticket","success","message"]
SIGNAL_HEADERS = ["timestamp","symbol","score","decision","confidence",
                   "ind_score","pa_score","ml_score"]


def _ensure_file(path: str, headers: list[str]):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(headers)


def log_signal(symbol: str, signal: dict):
    """Record a signal result to signal_log.csv and print to console."""
    _ensure_file(SIGNAL_LOG_FILE, SIGNAL_HEADERS)

    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dec = signal.get("decision", "?")
    scr = signal.get("score", 0)
    conf= signal.get("confidence", 0)

    # Console output
    colour = {"BUY": "\033[92m", "SELL": "\033[91m", "HOLD": "\033[93m"}
    reset  = "\033[0m"
    c      = colour.get(dec, "")
    print(f"  [{ts}] {symbol:<10} {c}{dec:<5}{reset} "
          f"score={scr:+.3f}  conf={conf:.1f}%  "
          f"(ind={signal.get('ind_score',0):+.2f} "
          f"pa={signal.get('pa_score',0):+.2f} "
          f"ml={signal.get('ml_score',0):+.2f})")

    with open(SIGNAL_LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            ts, symbol, scr, dec, conf,
            signal.get("ind_score", 0),
            signal.get("pa_score",  0),
            signal.get("ml_score",  0),
        ])


def log_trade(symbol: str, direction: str, lot: float,
              entry: float, sl: float, tp: float,
              ticket, success: bool, message: str):
    """Record an executed trade to trade_log.csv."""
    _ensure_file(TRADE_LOG_FILE, TRADE_HEADERS)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(TRADE_LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            ts, symbol, direction, lot, entry, sl, tp,
            ticket, success, message
        ])


def log_info(msg: str):
    """Generic info message with timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}] ℹ  {msg}")


def log_error(msg: str):
    """Red error message with timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\033[91m  [{ts}] ✗  {msg}\033[0m")


def print_session_summary():
    """Print a brief summary from the trade log."""
    if not os.path.exists(TRADE_LOG_FILE):
        print("  No trade log found.")
        return

    trades = []
    with open(TRADE_LOG_FILE) as f:
        reader = csv.DictReader(f)
        trades = list(reader)

    if not trades:
        print("  No trades recorded yet.")
        return

    total    = len(trades)
    success  = sum(1 for t in trades if t["success"] == "True")
    print(f"\n  Session summary: {total} orders sent, {success} filled successfully.")
