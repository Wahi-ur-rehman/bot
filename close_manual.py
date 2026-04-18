import MetaTrader5 as mt5
from trade_executor import _get_filling_mode, BOT_MAGIC

if not mt5.initialize():
    print("MT5 init failed")
    exit()

positions = mt5.positions_get()
if not positions:
    print("No open positions")
    mt5.shutdown()
    exit()

closed_count = 0
for p in positions:
    # 0.0 TP usually indicates manual trades, or magic number != BOT_MAGIC
    if p.tp == 0.0 or p.magic != BOT_MAGIC:
        tick = mt5.symbol_info_tick(p.symbol)
        close_type = mt5.ORDER_TYPE_SELL if p.type == 0 else mt5.ORDER_TYPE_BUY
        close_price = tick.bid if p.type == 0 else tick.ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": p.symbol,
            "volume": p.volume,
            "type": close_type,
            "position": p.ticket,
            "price": close_price,
            "deviation": 20,
            "magic": 0,
            "comment": "Close Manual",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": _get_filling_mode(p.symbol),
        }
        res = mt5.order_send(request)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✅ Closed manual {p.symbol} trade (Ticket #{p.ticket})")
            closed_count += 1
        else:
            print(f"❌ Failed to close {p.symbol}: {res.retcode if res else 'Unknown'}")

if closed_count == 0:
    print("No manual trades found to close.")
else:
    print(f"\nSuccessfully closed {closed_count} manual trades!")

mt5.shutdown()
