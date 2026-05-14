"""
trade_executor.py  ── v4 (Production)
────────────────────────────────────────
v4 upgrades over v3:
  ✔ partial_close()     — close 50% of position at 1R profit (lock gains)
  ✔ move_to_breakeven() — move SL to entry+1pip once profit ≥ threshold
  ✔ manage_open_trades() — called every cycle; handles BE + trailing stop
  ✔ _sl_distance_usd()  — used to feed PerformanceTracker for Kelly sizing
  ✔ monitor_positions() feeds PerformanceTracker on every closed trade
"""

import MetaTrader5 as mt5
import time
from datetime import datetime

from risk_manager import (
    update_trailing_stop,
    get_performance_tracker,
)


BOT_MAGIC = 202401

# Tracks tickets opened this session: {ticket: {symbol, direction, entry, sl, tp, lot, half_closed}}
_open_tickets: dict[int, dict] = {}


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════
def _get_filling_mode(symbol: str) -> int:
    info    = mt5.symbol_info(symbol)
    if not info:
        return mt5.ORDER_FILLING_IOC
    filling = getattr(info, "filling_mode", 0)
    if filling & 1:
        return mt5.ORDER_FILLING_FOK
    elif filling & 2:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN


def _sl_distance_usd(ticket_info: dict) -> float:
    """Estimate USD value of the SL distance for R-multiple tracking."""
    try:
        info     = mt5.symbol_info(ticket_info["symbol"])
        if info is None:
            return 0.0
        tick_val = info.trade_tick_value
        tick_sz  = info.trade_tick_size
        if tick_sz == 0:
            return 0.0
        sl_dist  = abs(ticket_info["entry"] - ticket_info["sl"])
        lot      = ticket_info["lot"]
        return (sl_dist / tick_sz) * tick_val * lot
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════════
# ORDER PLACEMENT
# ══════════════════════════════════════════════════════════════════════
def place_order(
    symbol   : str,
    direction: str,
    lot      : float,
    sl       : float,
    tp       : float,
    comment  : str = "ScalpBot_v4",
) -> dict:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return {"success": False, "ticket": None, "message": "No tick data"}

    order_type = mt5.ORDER_TYPE_BUY  if direction == "BUY"  else mt5.ORDER_TYPE_SELL
    price      = tick.ask            if direction == "BUY"  else tick.bid

    request = {
        "action"      : mt5.TRADE_ACTION_DEAL,
        "symbol"      : symbol,
        "volume"      : lot,
        "type"        : order_type,
        "price"       : price,
        "sl"          : sl,
        "tp"          : tp,
        "deviation"   : 20,
        "magic"       : BOT_MAGIC,
        "comment"     : comment,
        "type_time"   : mt5.ORDER_TIME_GTC,
        "type_filling": _get_filling_mode(symbol),
    }

    result = mt5.order_send(request)
    if result is None:
        return {"success": False, "ticket": None, "message": str(mt5.last_error())}

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        ticket = result.order
        _open_tickets[ticket] = {
            "symbol"     : symbol,
            "direction"  : direction,
            "entry"      : price,
            "sl"         : sl,
            "tp"         : tp,
            "lot"        : lot,
            "half_closed": False,   # track partial close state
            "be_moved"   : False,   # track breakeven state
        }
        msg = (f"[TRADE OPEN] {direction} {lot} {symbol} "
               f"@ {price:.5f}  SL={sl:.5f}  TP={tp:.5f}  #{ticket}")
        print(msg)
        return {"success": True, "ticket": ticket, "message": msg}
    else:
        msg = (f"[ERROR] Order failed for {symbol}: "
               f"retcode={result.retcode} ({result.comment})")
        print(msg)
        return {"success": False, "ticket": None, "message": msg}


# ══════════════════════════════════════════════════════════════════════
# PARTIAL CLOSE
# ══════════════════════════════════════════════════════════════════════
def partial_close(ticket: int, close_fraction: float = 0.5) -> dict:
    """
    Close `close_fraction` of the position volume at market.
    Used to lock in 50% profit once price reaches 1R.
    """
    position = mt5.positions_get(ticket=ticket)
    if not position:
        return {"success": False, "message": f"#{ticket} not found"}

    pos        = position[0]
    symbol     = pos.symbol
    direction  = pos.type
    full_vol   = pos.volume
    close_vol  = round(full_vol * close_fraction, 2)

    info = mt5.symbol_info(symbol)
    if info:
        step      = info.volume_step
        close_vol = round(round(close_vol / step) * step, 2)

    min_vol = info.volume_min if info else 0.01
    if close_vol < min_vol:
        return {"success": False, "message": f"#{ticket} volume too small to partial close"}

    close_type  = mt5.ORDER_TYPE_SELL if direction == 0 else mt5.ORDER_TYPE_BUY
    tick        = mt5.symbol_info_tick(symbol)
    close_price = tick.bid if direction == 0 else tick.ask

    request = {
        "action"      : mt5.TRADE_ACTION_DEAL,
        "symbol"      : symbol,
        "volume"      : close_vol,
        "type"        : close_type,
        "position"    : ticket,
        "price"       : close_price,
        "deviation"   : 20,
        "magic"       : BOT_MAGIC,
        "comment"     : "ScalpBot partial",
        "type_time"   : mt5.ORDER_TIME_GTC,
        "type_filling": _get_filling_mode(symbol),
    }

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        if ticket in _open_tickets:
            _open_tickets[ticket]["half_closed"] = True
            _open_tickets[ticket]["lot"] = round(full_vol - close_vol, 2)
        msg = f"[PARTIAL CLOSE] #{ticket} closed {close_vol} lots @ {close_price:.5f}"
        print(msg)
        return {"success": True, "message": msg}
    else:
        retcode = result.retcode if result else "N/A"
        return {"success": False, "message": f"[ERROR] Partial close #{ticket}: retcode={retcode}"}


# ══════════════════════════════════════════════════════════════════════
# BREAKEVEN MOVE
# ══════════════════════════════════════════════════════════════════════
def move_to_breakeven(ticket: int, entry: float, atr: float, direction: str) -> bool:
    """
    Shift SL to entry + 1 pip (BUY) or entry - 1 pip (SELL) once
    the trade has moved at least 1 ATR in our favour.
    Returns True if SL was modified.
    """
    position = mt5.positions_get(ticket=ticket)
    if not position:
        return False

    pos    = position[0]
    symbol = pos.symbol
    info   = mt5.symbol_info(symbol)
    digits = info.digits if info else 5
    pip    = 0.0001 if (info and "JPY" not in symbol) else 0.01

    if direction == "BUY":
        be_sl   = round(entry + pip, digits)
        if pos.sl >= be_sl:
            return False   # already at or beyond breakeven
        # only move if price is sufficiently in profit
        tick = mt5.symbol_info_tick(symbol)
        if tick is None or tick.bid < entry + atr:
            return False
    else:
        be_sl = round(entry - pip, digits)
        if pos.sl <= be_sl:
            return False
        tick  = mt5.symbol_info_tick(symbol)
        if tick is None or tick.ask > entry - atr:
            return False

    request = {
        "action"  : mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "sl"      : be_sl,
        "tp"      : pos.tp,
    }
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        if ticket in _open_tickets:
            _open_tickets[ticket]["be_moved"] = True
        print(f"[BE MOVE] #{ticket} {symbol} SL → {be_sl:.5f} (breakeven)")
        return True
    return False


# ══════════════════════════════════════════════════════════════════════
# TRADE MANAGEMENT — called every cycle
# ══════════════════════════════════════════════════════════════════════
def manage_open_trades() -> None:
    """
    For each bot-opened trade, every cycle:
      1. Partial close at 1R if not already done
      2. Move SL to breakeven once 1 ATR has moved in our favour
      3. Trail the stop by 1 ATR
    Requires current price and ATR from MT5 live data.
    """
    if not _open_tickets:
        return

    live = mt5.positions_get()
    if not live:
        return
    live_map = {p.ticket: p for p in live}

    for ticket, info in list(_open_tickets.items()):
        pos = live_map.get(ticket)
        if pos is None:
            continue

        symbol    = info["symbol"]
        direction = info["direction"]
        entry     = info["entry"]
        sl_orig   = info["sl"]

        # Get live ATR from symbol info (approximate — use last candle ATR)
        # We use spread + point as a minimum ATR proxy if full df not available
        sym_info  = mt5.symbol_info(symbol)
        tick      = mt5.symbol_info_tick(symbol)
        if sym_info is None or tick is None:
            continue

        current_price = tick.bid if direction == "BUY" else tick.ask
        # ATR proxy: use 10x average spread as minimum
        atr_approx = max(sym_info.point * 50, abs(entry - sl_orig) / 1.5)

        # 1. Partial close at 1R
        if not info["half_closed"]:
            r1_level = (entry + abs(entry - sl_orig)) if direction == "BUY" else (
                entry - abs(entry - sl_orig))
            if (direction == "BUY"  and current_price >= r1_level) or \
               (direction == "SELL" and current_price <= r1_level):
                partial_close(ticket, close_fraction=0.5)

        # 2. Move to breakeven
        if not info["be_moved"]:
            move_to_breakeven(ticket, entry, atr_approx, direction)

        # 3. Trail stop (only after breakeven is set)
        if info["be_moved"]:
            update_trailing_stop(ticket, direction, current_price, atr_approx)


# ══════════════════════════════════════════════════════════════════════
# MONITOR (TP/SL detection)
# ══════════════════════════════════════════════════════════════════════
def monitor_positions() -> list[dict]:
    """
    Compare bot's known open tickets against MT5 current positions.
    Any ticket that disappeared was closed by MT5 (TP/SL hit).
    Feeds PerformanceTracker for Kelly sizing.
    Returns list of closed trade outcomes.
    """
    if not _open_tickets:
        return []

    live = mt5.positions_get()
    live_tickets = {p.ticket for p in live} if live else set()

    tracker = get_performance_tracker()
    closed  = []

    for ticket, info in list(_open_tickets.items()):
        if ticket not in live_tickets:
            outcome = _check_deal_outcome(ticket, info)
            # Feed Kelly tracker
            sl_usd = _sl_distance_usd(info)
            tracker.record(outcome["result"], outcome["pnl"], sl_usd)

            closed.append(outcome)
            del _open_tickets[ticket]
            print(f"  [CLOSED] #{ticket} {info['symbol']} {info['direction']} "
                  f"→ {outcome['result']}  P&L≈{outcome['pnl']:+.2f}")

    return closed


def _check_deal_outcome(ticket: int, info: dict) -> dict:
    """Look up closing deal in MT5 history to determine TP/SL/manual."""
    to_time = int(datetime.now().timestamp()) + 60
    deals   = mt5.history_deals_get(0, to_time)
    result  = "UNKNOWN"
    pnl     = 0.0

    if deals:
        matching = [d for d in deals
                    if d.position_id == ticket and d.entry == 1]
        if matching:
            deal    = matching[-1]
            pnl     = deal.profit
            comment = (deal.comment or "").lower()
            if "tp" in comment or pnl > 0:
                result = "TP_HIT"
            elif "sl" in comment or pnl < 0:
                result = "SL_HIT"
            else:
                result = "MANUAL_CLOSE"

    return {
        "ticket"   : ticket,
        "symbol"   : info["symbol"],
        "direction": info["direction"],
        "result"   : result,
        "pnl"      : pnl,
    }


# ══════════════════════════════════════════════════════════════════════
# CLOSE UTILITIES
# ══════════════════════════════════════════════════════════════════════
def close_position(ticket: int) -> dict:
    position = mt5.positions_get(ticket=ticket)
    if not position:
        _open_tickets.pop(ticket, None)
        return {"success": False, "message": f"#{ticket} not found (already closed?)"}

    pos         = position[0]
    symbol      = pos.symbol
    direction   = pos.type
    volume      = pos.volume
    tick        = mt5.symbol_info_tick(symbol)
    close_type  = mt5.ORDER_TYPE_SELL if direction == 0 else mt5.ORDER_TYPE_BUY
    close_price = tick.bid if direction == 0 else tick.ask

    request = {
        "action"      : mt5.TRADE_ACTION_DEAL,
        "symbol"      : symbol,
        "volume"      : volume,
        "type"        : close_type,
        "position"    : ticket,
        "price"       : close_price,
        "deviation"   : 20,
        "magic"       : BOT_MAGIC,
        "comment"     : "ScalpBot close",
        "type_time"   : mt5.ORDER_TIME_GTC,
        "type_filling": _get_filling_mode(symbol),
    }

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        _open_tickets.pop(ticket, None)
        msg = f"[TRADE CLOSE] #{ticket} closed @ {close_price:.5f}"
        print(msg)
        return {"success": True, "message": msg}
    else:
        retcode = result.retcode if result else "N/A"
        return {"success": False, "message": f"[ERROR] Cannot close #{ticket}: retcode={retcode}"}


def close_all_positions() -> list[dict]:
    positions = mt5.positions_get()
    results   = []
    if not positions:
        print("[INFO] No open positions.")
        return results
    for pos in positions:
        if pos.magic == BOT_MAGIC:
            res = close_position(pos.ticket)
            results.append(res)
            time.sleep(0.1)
    return results


# ══════════════════════════════════════════════════════════════════════
# SIGNAL EXECUTION
# ══════════════════════════════════════════════════════════════════════
def execute_signals(signals: list[dict]) -> list[dict]:
    results = []
    for sig in signals:
        if sig.get("decision") in ("BUY", "SELL"):
            res = place_order(
                symbol    = sig["symbol"],
                direction = sig["decision"],
                lot       = sig["lot"],
                sl        = sig["sl"],
                tp        = sig["tp"],
            )
            results.append({**sig, **res})
            time.sleep(0.15)  # small delay between orders
    return results


# ══════════════════════════════════════════════════════════════════════
# DISPLAY
# ══════════════════════════════════════════════════════════════════════
def print_open_positions():
    positions = mt5.positions_get()
    if not positions:
        print("  No open positions.")
        return

    print(f"\n{'─'*72}")
    print(f"  {'Symbol':<10} {'Type':<6} {'Lot':<6} {'Entry':>10} "
          f"{'SL':>10} {'TP':>10} {'P&L':>8} {'BE':>4}")
    print(f"{'─'*72}")

    for p in positions:
        direction = "BUY" if p.type == 0 else "SELL"
        info      = _open_tickets.get(p.ticket, {})
        be_tag    = "✓" if info.get("be_moved") else " "
        print(f"  {p.symbol:<10} {direction:<6} {p.volume:<6.2f} "
              f"{p.price_open:>10.5f} {p.sl:>10.5f} {p.tp:>10.5f} "
              f"{p.profit:>+8.2f} {be_tag:>4}")

    total_pnl = sum(p.profit for p in positions)
    print(f"{'─'*72}")
    print(f"  {'Total P&L':>59}  {total_pnl:>+8.2f}\n")
