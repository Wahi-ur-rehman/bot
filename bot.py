"""
bot.py  ── v4 (Production Grade)
─────────────────────────────────
THE MAIN FILE.  Run:  python bot.py

v4 upgrades over v3:
  ✔ Circuit breaker       — halts trading at 3% daily drawdown
  ✔ Spread monitor        — skips symbols with spread > 1.5× rolling avg
  ✔ Session filter        — only trades London + NY liquid hours
  ✔ Regime filter         — skips RANGING and VOLATILE markets
  ✔ ADX gate              — skips if ADX < 20
  ✔ Multi-symbol ML       — trains LightGBM on ALL symbols combined
  ✔ Parallel fetch        — 8-thread parallel MT5 data pulls
  ✔ Adaptive SL/TP        — regime-based multipliers (trend/range/vol)
  ✔ Kelly position sizing — 25% fractional Kelly once 10+ trades logged
  ✔ Partial close at 1R   — locks 50% profit, moves SL to breakeven
  ✔ Trailing stop         — trails SL by 1 ATR after breakeven set
  ✔ Performance tracker   — rolling win rate / avg R printed each cycle
  ✔ Scan interval 30s     — faster than 60s for M5 scalping
  ✔ Confidence gate 35%   — raised from 20% to filter noise
  ✔ Streak required 2     — 2 consecutive same-direction cycles
"""

import time
import sys
import io
from datetime import datetime, time as dtime
from collections import deque

if isinstance(sys.stdout, io.TextIOWrapper) and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from mt5_connector   import connect, disconnect, fetch_all_symbols
from analysis_engine import (
    combined_signal, MLSignal, SignalPersistence,
    compute_indicators, MarketRegimeDetector,
)
from risk_manager import (
    calculate_sl_tp, can_open_trade, MAX_OPEN_TRADES,
    get_performance_tracker,
)
from trade_executor import (
    execute_signals, print_open_positions,
    monitor_positions, manage_open_trades,
)
from logger import log_signal, log_trade, log_info, log_error, print_session_summary
from ai_judge import AIJudge
import MetaTrader5 as mt5
import numpy as np


# ══════════════════════════════════════════════════════════════════════
# CONFIG — tuned for real M5 scalping
# ══════════════════════════════════════════════════════════════════════
SCAN_INTERVAL    = 30          # 30s scan cycle (was 60s)
RETRAIN_EVERY    = 30          # retrain ML every 30 cycles (~15 min)
MIN_CONFIDENCE   = 35.0        # noise gate — raised from 20%
STREAK_REQUIRED  = 2           # 2 consecutive agreeing signals
MAX_DAILY_LOSS   = 0.03        # 3% daily drawdown circuit breaker
MAX_SPREAD_MULT  = 1.5         # skip if spread > 1.5× rolling average
MIN_ADX          = 20          # skip ranging markets

# AI Integration
AI_CONFIRMATION  = True        # Set to True to enable AI confirmation layer
AI_PROVIDER      = "gemini"    # "gemini", "groq", or "openrouter"
AI_API_KEY       = ""          # ⬅️ INSERT YOUR API KEY HERE

# Trading sessions (UTC) — only enter during high liquidity
TRADING_SESSIONS = {
    "london":  (dtime(7, 0),   dtime(16, 0)),
    "ny":      (dtime(12, 0),  dtime(21, 0)),
    "overlap": (dtime(12, 0),  dtime(16, 0)),   # best for scalping
}


# ══════════════════════════════════════════════════════════════════════
# SPREAD MONITOR
# ══════════════════════════════════════════════════════════════════════
class SpreadMonitor:
    """Track rolling average spread per symbol; skip when too wide."""

    def __init__(self, window: int = 100):
        self._history: dict[str, deque] = {}
        self._window = window

    def update(self, symbol: str) -> int | None:
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        spread = info.spread
        if symbol not in self._history:
            self._history[symbol] = deque(maxlen=self._window)
        self._history[symbol].append(spread)
        return spread

    def is_acceptable(self, symbol: str) -> tuple[bool, str]:
        spread = self.update(symbol)
        if spread is None:
            return False, "no info"
        hist = self._history[symbol]
        if len(hist) < 20:
            return True, f"sp={spread}"  # warming up — allow
        avg = float(np.mean(hist))
        if avg > 0 and spread > avg * MAX_SPREAD_MULT:
            return False, f"sp={spread}>{MAX_SPREAD_MULT}×avg({avg:.0f})"
        return True, f"sp={spread}"


# ══════════════════════════════════════════════════════════════════════
# SESSION FILTER
# ══════════════════════════════════════════════════════════════════════
def is_trading_session() -> tuple[bool, str]:
    """Check if current UTC time falls within a liquid trading session."""
    now_utc = datetime.utcnow().time()
    for name, (start, end) in TRADING_SESSIONS.items():
        if start <= now_utc <= end:
            return True, name
    return False, "off-session"


# ══════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER
# ══════════════════════════════════════════════════════════════════════
class CircuitBreaker:
    """Hard daily drawdown limit — halts all trading when tripped."""

    def __init__(self, max_daily_loss_pct: float):
        self._max_loss      = max_daily_loss_pct
        self._start_balance = self._get_balance()
        self._day           = datetime.utcnow().date()
        self._tripped       = False

    @staticmethod
    def _get_balance() -> float:
        info = mt5.account_info()
        return info.balance if info else 0.0

    def check(self) -> tuple[bool, str]:
        today = datetime.utcnow().date()
        if today != self._day:
            # New day — reset
            self._start_balance = self._get_balance()
            self._day           = today
            self._tripped       = False

        if self._tripped:
            return False, "CIRCUIT BREAKER tripped for today"

        current = self._get_balance()
        if self._start_balance <= 0:
            return True, "no balance data"

        dd = (self._start_balance - current) / self._start_balance

        if dd >= self._max_loss:
            self._tripped = True
            log_error(f"🚨 CIRCUIT BREAKER: Daily DD {dd*100:.2f}% ≥ {self._max_loss*100:.0f}%")
            return False, f"daily DD {dd*100:.2f}%"

        return True, f"DD {dd*100:.2f}%"


# ══════════════════════════════════════════════════════════════════════
# MULTI-SYMBOL ML TRAINING
# ══════════════════════════════════════════════════════════════════════
def train_ml_multi(ml: MLSignal, market_data: dict) -> bool:
    """Train ML model on combined data from ALL symbols."""
    success = ml.train_multi(market_data)
    if success:
        log_info(f"✅ ML trained on {len(market_data)} symbols (multi-symbol).")
    else:
        log_info("⚠ ML training skipped — insufficient data.")
    return success


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════
def _count_open_positions() -> int:
    positions = mt5.positions_get()
    return len(positions) if positions else 0


# ══════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════
def run():
    print("\n" + "═" * 68)
    print("   SCALPING BOT v4  │  PRODUCTION GRADE  │  starting …")
    print("   LightGBM · Regime Filter · Kelly Sizing · Circuit Breaker")
    print("═" * 68 + "\n")

    if not connect():
        print("[FATAL] Cannot connect to MT5.")
        sys.exit(1)

    # ─── Initialize subsystems ─────────────────────────────────────
    persistence    = SignalPersistence(required_streak=STREAK_REQUIRED)
    spread_monitor = SpreadMonitor()
    breaker        = CircuitBreaker(MAX_DAILY_LOSS)
    ml             = MLSignal()
    tracker        = get_performance_tracker()

    # ─── Initial ML training on all symbols ────────────────────────
    log_info("Fetching initial data for multi-symbol ML training …")
    initial_data = fetch_all_symbols()

    if initial_data:
        train_ml_multi(ml, initial_data)
    else:
        log_error("No data from MT5. Check symbols and connection.")

    # ─── Initialize AI Judge ───────────────────────────────────────
    ai_judge = None
    if AI_CONFIRMATION:
        try:
            ai_judge = AIJudge(provider=AI_PROVIDER, api_key=AI_API_KEY)
            log_info(f"✅ AI Judge initialized ({AI_PROVIDER})")
        except Exception as e:
            log_error(f"Failed to initialize AI Judge: {e}")

    # Session stats
    session_tp     = 0
    session_sl     = 0
    session_pnl    = 0.0
    cycle          = 0
    news_sentiment = {}

    try:
        while True:
            cycle += 1
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # ── Circuit breaker ──────────────────────────────────────
            cb_ok, cb_msg = breaker.check()
            if not cb_ok:
                log_error(f"⛔ Trading halted: {cb_msg}")
                time.sleep(60)
                continue

            # ── Session filter ───────────────────────────────────────
            in_session, sess_name = is_trading_session()

            print(f"\n{'─' * 68}")
            print(f"  Cycle #{cycle}  │  {now}  │  Session: {sess_name.upper()}")
            print(f"  TP:{session_tp}  SL:{session_sl}  "
                  f"P&L:{session_pnl:+.2f}  │  {cb_msg}")
            print(f"{'─' * 68}")

            if not in_session:
                log_info("⏸ Off-session — monitoring only, no new entries.")

            # ── 1. Manage existing trades (partial close / trailing) ──
            manage_open_trades()

            # ── 2. Detect closed positions (TP/SL hit) ───────────────
            closed_this_cycle = monitor_positions()
            for c in closed_this_cycle:
                session_pnl += c["pnl"]
                if c["result"] == "TP_HIT":
                    session_tp += 1
                    log_info(f"✅ TP: {c['symbol']} {c['direction']}  +{c['pnl']:.2f}")
                elif c["result"] == "SL_HIT":
                    session_sl += 1
                    log_info(f"❌ SL: {c['symbol']} {c['direction']}  {c['pnl']:.2f}")
                else:
                    log_info(f"ℹ {c['result']}: {c['symbol']}  {c['pnl']:.2f}")

            if _count_open_positions() == 0 and closed_this_cycle:
                persistence.reset_all()
                log_info("Positions flat — persistence reset.")

            # ── 3. Fetch market data (parallel) ──────────────────────
            market_data = fetch_all_symbols()
            if not market_data:
                log_error("No market data — skipping cycle.")
                time.sleep(SCAN_INTERVAL)
                continue

            # ── 4. Periodic ML retrain ───────────────────────────────
            if cycle % RETRAIN_EVERY == 0:
                log_info("🔄 Retraining ML (multi-symbol) …")
                train_ml_multi(ml, market_data)

            # ── 5. Print Kelly tracker stats ─────────────────────────
            if tracker.trade_count > 0 and cycle % 10 == 0:
                tracker.print_stats()

            # ── 5b. Periodic News Sentiment check ────────────────────
            if ai_judge and cycle % 60 == 0: # Every ~30 mins
                currencies = list(set([s[:3] for s in market_data.keys()] + [s[3:6] for s in market_data.keys() if len(s)==6]))
                from ai_judge import get_news_sentiment
                news_sentiment = get_news_sentiment(ai_judge, currencies)

            # ── 6. Analyse each symbol ───────────────────────────────
            trade_candidates   = []
            symbols_this_cycle = set()

            print(f"\n  {'Symbol':<10} {'Dec':<5} {'Conf':>6} {'Regime':<10} "
                  f"{'ADX':>5}  {'Spread':<14}")
            print(f"  {'─' * 62}")

            for symbol, df in market_data.items():
                signal = combined_signal(df, ml)
                log_signal(symbol, signal)

                dec    = signal["decision"]
                conf   = signal["confidence"]
                adx    = signal.get("adx", 0)
                regime = signal.get("regime", "NEUTRAL")

                # Spread check
                sp_ok, sp_msg = spread_monitor.is_acceptable(symbol)

                # Coloured output
                dec_colors = {
                    "BUY": "\033[92m", "SELL": "\033[91m", "HOLD": "\033[93m"
                }
                reg_colors = {
                    "TRENDING": "\033[92m", "RANGING": "\033[91m",
                    "VOLATILE": "\033[95m", "NEUTRAL": "\033[93m",
                }
                reset = "\033[0m"
                dc    = dec_colors.get(dec, "")
                rc    = reg_colors.get(regime, "")

                print(f"  {symbol:<10} {dc}{dec:<5}{reset} {conf:>5.1f}% "
                      f"{rc}{regime:<10}{reset} "
                      f"{adx:>5.1f}  {sp_msg:<14}")

                # ── ALL FILTERS MUST PASS ──────────────────────────
                if not in_session:
                    continue
                if dec == "HOLD":
                    continue
                if conf < MIN_CONFIDENCE:
                    continue
                if regime in ("RANGING", "VOLATILE"):
                    log_info(f"  ⏸ {symbol}: skip {regime} regime")
                    continue
                if adx < MIN_ADX:
                    continue
                if not sp_ok:
                    log_info(f"  ⏸ {symbol}: spread too wide ({sp_msg})")
                    continue
                if symbol in symbols_this_cycle:
                    continue

                # ── Persistence ────────────────────────────────────
                confirmed = persistence.confirm(symbol, dec)
                if not confirmed:
                    log_info(f"  ⏳ {symbol}: {dec} {conf:.1f}% — streak pending")
                    continue

                ok, reason = can_open_trade(symbol)
                if not ok:
                    log_info(f"  ⏸ {symbol}: {reason}")
                    continue

                # ── Calculate SL/TP with regime awareness ──────────
                sl_tp = calculate_sl_tp(symbol, dec, df, regime=regime)

                # ── AI Confirmation Layer ──────────────────────────
                if ai_judge:
                    from ai_judge import AI_MIN_AGREE
                    verdict = ai_judge.confirm_signal(symbol, signal, df)
                    if AI_MIN_AGREE and not verdict["confirmed"]:
                        log_info(f"  🤖 {symbol}: AI vetoed trade — {verdict['reason']}")
                        continue

                trade_candidates.append({
                    "symbol"    : symbol,
                    "decision"  : dec,
                    "lot"       : sl_tp["lot"],
                    "sl"        : sl_tp["sl"],
                    "tp"        : sl_tp["tp"],
                    "entry"     : sl_tp["entry"],
                    "confidence": conf,
                    "regime"    : regime,
                    "ai_reason" : verdict["reason"] if ai_judge else None
                })
                symbols_this_cycle.add(symbol)

            # ── 7. Execute trades (sorted by confidence) ─────────────
            if trade_candidates:
                trade_candidates.sort(key=lambda x: x["confidence"], reverse=True)
                # Cap trades this cycle to available capacity
                capacity = max(1, MAX_OPEN_TRADES - _count_open_positions())
                trade_candidates = trade_candidates[:capacity]

                log_info(f"📤 Placing {len(trade_candidates)} trade(s) …")
                results = execute_signals(trade_candidates)
                for res in results:
                    log_trade(
                        symbol    = res["symbol"],
                        direction = res["decision"],
                        lot       = res["lot"],
                        entry     = res["entry"],
                        sl        = res["sl"],
                        tp        = res["tp"],
                        ticket    = res.get("ticket"),
                        success   = res.get("success", False),
                        message   = res.get("message", ""),
                    )
            else:
                log_info("No qualifying signals this cycle.")

            # ── 8. Show positions ────────────────────────────────────
            print("\n  Open positions:")
            print_open_positions()

            log_info(f"⏱ Next scan in {SCAN_INTERVAL}s …")
            time.sleep(SCAN_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n  [STOP] Bot stopped by user.")
        print(f"  Session: TP={session_tp}  SL={session_sl}  "
              f"Net P&L={session_pnl:+.2f}")
        if tracker.trade_count > 0:
            tracker.print_stats()
        print_session_summary()

    finally:
        disconnect()
        print("\n  Goodbye.\n")


if __name__ == "__main__":
    run()
