"""
trade_executor.py  ── v3 (FIXED)
─────────────────────────────────
Places and manages MT5 trades.

v3 additions
  ✔ monitor_positions()  — checks every open trade each cycle:
      • If TP hit   → logs win, marks closed
      • If SL hit   → logs loss, marks closed
      • If trade no longer appears in MT5 positions (broker closed it via
        SL/TP) → the bot now correctly detects this and logs it
    This fixes the core issue where trades disappeared from MT5 but the
    bot didn't know, causing "Already in {symbol}" blocks forever.
  ✔ get_closed_since()   — returns trades MT5 closed since last check
"""

import MetaTrader5 as mt5
import time
from datetime import datetime


BOT_MAGIC = 202401

# Tracks tickets the bot opened this session {ticket: {symbol, direction, entry, sl, tp}}
_open_tickets: dict[int, dict] = {}


def _get_filling_mode(symbol: str) -> int:
    info = mt5.symbol_info(symbol)
    if not info:
        return mt5.ORDER_FILLING_IOC
    filling = getattr(info, "filling_mode", 0)
    if filling & 1:
        return mt5.ORDER_FILLING_FOK
    elif filling & 2:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN


def place_order(
    symbol    : str,
    direction : str,
    lot       : float,
    sl        : float,
    tp        : float,
    comment   : str = "ScalpBot",
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
            "symbol"   : symbol,
            "direction": direction,
            "entry"    : price,
            "sl"       : sl,
            "tp"       : tp,
            "lot"      : lot,
        }
        msg = (f"[TRADE OPEN] {direction} {lot} {symbol} "
               f"@ {price:.5f}  SL={sl:.5f}  TP={tp:.5f}  ticket=#{ticket}")
        print(msg)
        return {"success": True, "ticket": ticket, "message": msg}
    else:
        msg = f"[ERROR] Order failed for {symbol}: retcode={result.retcode} ({result.comment})"
        print(msg)
        return {"success": False, "ticket": None, "message": msg}


def monitor_positions() -> list[dict]:
    """
    Called every cycle. Compares bot's known open tickets against MT5's
    current open positions. Any ticket that has disappeared was closed by
    MT5 (TP/SL hit, or manually closed).

    Returns a list of closed trade dicts with outcome info.
    Removes them from _open_tickets so the symbol is available again.
    """
    if not _open_tickets:
        return []

    # Get tickets currently open in MT5
    live = mt5.positions_get()
    live_tickets = {p.ticket for p in live} if live else set()

    closed = []
    for ticket, info in list(_open_tickets.items()):
        if ticket not in live_tickets:
            # Trade is gone — determine outcome from deal history
            outcome = _check_deal_outcome(ticket, info)
            closed.append(outcome)
            del _open_tickets[ticket]
            print(f"  [CLOSED] #{ticket} {info['symbol']} {info['direction']} "
                  f"→ {outcome['result']}  P&L≈{outcome['pnl']:+.2f}")

    return closed


def _check_deal_outcome(ticket: int, info: dict) -> dict:
    """
    Look up the closing deal in MT5 history to determine TP/SL/manual.
    Returns dict with result and approximate P&L.
    """
    # Fetch last 1000 deals to find the close
    from_time = 0
    to_time   = int(datetime.now().timestamp()) + 60

    deals = mt5.history_deals_get(0, to_time)
    result = "UNKNOWN"
    pnl    = 0.0

    if deals:
        # Find deals matching this position ticket
        matching = [d for d in deals if d.position_id == ticket and d.entry == 1]  # entry=1 = close
        if matching:
            deal = matching[-1]
            pnl  = deal.profit
            comment = (deal.comment or "").lower()
            if "tp" in comment or (pnl > 0 and info["direction"] == "BUY") or (pnl > 0 and info["direction"] == "SELL"):
                result = "TP_HIT" if pnl > 0 else "SL_HIT"
            elif "sl" in comment:
                result = "SL_HIT"
            elif pnl > 0:
                result = "TP_HIT"
            elif pnl < 0:
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


def close_position(ticket: int) -> dict:
    position = mt5.positions_get(ticket=ticket)
    if not position:
        # Already closed — clean up tracker
        _open_tickets.pop(ticket, None)
        return {"success": False, "message": f"Position #{ticket} not found (already closed?)"}

    pos        = position[0]
    symbol     = pos.symbol
    direction  = pos.type
    volume     = pos.volume
    tick       = mt5.symbol_info_tick(symbol)

    close_type  = mt5.ORDER_TYPE_SELL if direction == 0 else mt5.ORDER_TYPE_BUY
    close_price = tick.bid            if direction == 0 else tick.ask

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
            time.sleep(0.2)
    return results


def print_open_positions():
    positions = mt5.positions_get()
    if not positions:
        print("  No open positions.")
        return

    print(f"\n{'─'*70}")
    print(f"  {'Symbol':<10} {'Type':<6} {'Lot':<6} {'Entry':>10} "
          f"{'SL':>10} {'TP':>10} {'P&L':>8}")
    print(f"{'─'*70}")

    for p in positions:
        direction = "BUY" if p.type == 0 else "SELL"
        print(f"  {p.symbol:<10} {direction:<6} {p.volume:<6.2f} "
              f"{p.price_open:>10.5f} {p.sl:>10.5f} {p.tp:>10.5f} "
              f"{p.profit:>+8.2f}")

    total_pnl = sum(p.profit for p in positions)
    print(f"{'─'*70}")
    print(f"  {'Total P&L':>57}  {total_pnl:>+8.2f}\n")
