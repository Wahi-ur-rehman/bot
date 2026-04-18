"""
bot.py  ── v3 (FIXED)
──────────────────────
THE MAIN FILE.  Run:  python bot.py

v3 core fixes
  ✔ monitor_positions() called every cycle
      → detects when MT5 closes trades via SL/TP (the main bug)
      → logs TP_HIT / SL_HIT / MANUAL_CLOSE outcomes
  ✔ persistence.reset_all() called when all positions close
      → bot immediately re-evaluates and places new trades
      → fixes the "stopped placing trades after manual close" bug
  ✔ MIN_CONFIDENCE = 25.0  (balanced — not so low it takes noise, not so
      high it never fires)
  ✔ STREAK_REQUIRED = 2    (2 consecutive same-direction cycles to confirm)
  ✔ Cycle summary printed: shows how many trades were TP/SL'd
"""

import time
import sys
import io
from datetime import datetime

if isinstance(sys.stdout, io.TextIOWrapper) and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from mt5_connector   import connect, disconnect, fetch_all_symbols
from analysis_engine import combined_signal, MLSignal, SignalPersistence, compute_indicators
from risk_manager    import calculate_sl_tp, can_open_trade, MAX_OPEN_TRADES
from trade_executor  import execute_signals, print_open_positions, monitor_positions
from logger          import log_signal, log_trade, log_info, log_error, print_session_summary
import MetaTrader5 as mt5


# ── Settings ──────────────────────────────────────────────────────────────────
SCAN_INTERVAL   = 60
RETRAIN_EVERY   = 20
MIN_CONFIDENCE  = 20.0   # fires on moderate signals — ADX+trend filters do the quality work
STREAK_REQUIRED = 1      # fire on first confirmed signal; streak=2 was killing entries


def _best_training_symbol(market_data: dict) -> tuple:
    return max(market_data.items(), key=lambda kv: len(kv[1]))


def _count_open_positions() -> int:
    positions = mt5.positions_get()
    return len(positions) if positions else 0


def run():
    print("\n" + "═" * 60)
    print("   SCALPING BOT v3  |  MT5  |  starting …")
    print("═" * 60 + "\n")

    if not connect():
        print("[FATAL] Cannot connect to MT5.")
        sys.exit(1)

    persistence = SignalPersistence(required_streak=STREAK_REQUIRED)

    ml = MLSignal()
    log_info("Fetching initial data to train ML model …")
    initial_data = fetch_all_symbols()

    if initial_data:
        sym, df = _best_training_symbol(initial_data)
        if ml.train(compute_indicators(df)):
            log_info(f"ML trained on {sym}.")
        else:
            log_info("ML training skipped — using indicators + price action only.")
    else:
        log_error("No data from MT5. Check symbols and connection.")

    # Session stats
    session_tp   = 0
    session_sl   = 0
    session_pnl  = 0.0

    cycle = 0
    try:
        while True:
            cycle += 1
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n{'─' * 60}")
            print(f"  Cycle #{cycle}  |  {now}  |  TP:{session_tp}  SL:{session_sl}  PnL:{session_pnl:+.2f}")
            print(f"{'─' * 60}")

            # ── 1. Check if any positions were closed by MT5 (TP/SL) ────────
            closed_this_cycle = monitor_positions()
            for c in closed_this_cycle:
                session_pnl += c["pnl"]
                if c["result"] == "TP_HIT":
                    session_tp += 1
                    log_info(f"✅ TP HIT: {c['symbol']} {c['direction']}  +{c['pnl']:.2f}")
                elif c["result"] == "SL_HIT":
                    session_sl += 1
                    log_info(f"❌ SL HIT: {c['symbol']} {c['direction']}  {c['pnl']:.2f}")
                else:
                    log_info(f"ℹ CLOSED: {c['symbol']} {c['direction']} ({c['result']})  {c['pnl']:.2f}")

            # ── 2. If all positions closed → reset persistence so bot can re-enter ──
            if _count_open_positions() == 0 and closed_this_cycle:
                log_info("All positions closed — resetting signal tracker for fresh entries.")
                persistence.reset_all()

            # ── 3. Fetch market data ─────────────────────────────────────────
            market_data = fetch_all_symbols()
            if not market_data:
                log_error("No market data — skipping cycle.")
                time.sleep(SCAN_INTERVAL)
                continue

            # ── 4. Periodic ML retrain ───────────────────────────────────────
            if cycle % RETRAIN_EVERY == 0:
                log_info("Retraining ML …")
                sym, df = _best_training_symbol(market_data)
                ml.train(compute_indicators(df))

            # ── 5. Analyse each symbol ───────────────────────────────────────
            trade_candidates  = []
            symbols_this_cycle = set()

            print(f"\n  {'Symbol':<10} {'Dec':<5} {'Conf':>6} {'Score':>7} "
                  f"{'Trend':<5} {'ADX':>5}  {'Ind':>6} {'PA':>6} {'ML':>6}")
            print(f"  {'─' * 65}")

            for symbol, df in market_data.items():
                signal = combined_signal(df, ml)
                log_signal(symbol, signal)

                dec   = signal["decision"]
                conf  = signal["confidence"]
                adx   = signal.get("adx", 0)
                trend = signal.get("trend", "?")

                colour = {"BUY": "\033[92m", "SELL": "\033[91m", "HOLD": "\033[93m"}
                reset  = "\033[0m"
                c      = colour.get(dec, "")
                print(f"  {symbol:<10} {c}{dec:<5}{reset} {conf:>5.1f}% "
                      f"{signal['score']:>+7.3f} "
                      f"{trend:<5} {adx:>5.1f}  "
                      f"{signal['ind_score']:>+6.2f} "
                      f"{signal['pa_score']:>+6.2f} "
                      f"{signal['ml_score']:>+6.2f}")

                confirmed = persistence.confirm(symbol, dec)

                if (dec != "HOLD"
                        and conf >= MIN_CONFIDENCE
                        and confirmed
                        and symbol not in symbols_this_cycle):

                    ok, reason = can_open_trade(symbol)
                    if ok:
                        sl_tp = calculate_sl_tp(symbol, dec, df)
                        trade_candidates.append({
                            "symbol"    : symbol,
                            "decision"  : dec,
                            "lot"       : sl_tp["lot"],
                            "sl"        : sl_tp["sl"],
                            "tp"        : sl_tp["tp"],
                            "entry"     : sl_tp["entry"],
                            "confidence": conf,
                        })
                        symbols_this_cycle.add(symbol)
                    else:
                        log_info(f"Skip {symbol}: {reason}")
                elif dec != "HOLD" and conf >= MIN_CONFIDENCE and not confirmed:
                    log_info(f"  ⏳ {symbol}: signal {dec} conf={conf:.1f}% — waiting for streak confirmation")

            # ── 6. Place trades ──────────────────────────────────────────────
            if trade_candidates:
                trade_candidates.sort(key=lambda x: x["confidence"], reverse=True)
                log_info(f"Placing {len(trade_candidates)} trade(s) …")
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

            # ── 7. Show positions ────────────────────────────────────────────
            print("\n  Open positions:")
            print_open_positions()

            log_info(f"Next scan in {SCAN_INTERVAL}s …")
            time.sleep(SCAN_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n  [STOP] Bot stopped.")
        print(f"  Session: TP={session_tp}  SL={session_sl}  Net P&L={session_pnl:+.2f}")
        print_session_summary()

    finally:
        disconnect()
        print("\n  Goodbye.\n")


if __name__ == "__main__":
    run()
