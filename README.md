# Scalping Bot v3 - MetaTrader 5

A high-frequency scalping bot for MetaTrader 5 (MT5) that combines technical indicators, price action analysis, and machine learning to execute trades.

## Features
- **MT5 Integration**: Automated connection and trade execution.
- **Confluence Engine**: Combines RSI, MACD, EMA trends, and ADX for high-probability signals.
- **Machine Learning**: Uses a `RandomForestClassifier` to predict price movement direction.
- **Price Action**: Detects candlestick patterns (Hammer, Engulfing, Morning/Evening Star).
- **Risk Management**: Automated SL/TP calculation and position monitoring.
- **Persistence Tracker**: Requires signal confirmation over multiple cycles to avoid noise.

## Installation

1. Install MetaTrader 5.
2. Ensure Python 3.8+ is installed.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Open your MT5 terminal and log in to your trading account.
2. Run the bot:
   ```bash
   python bot.py
   ```

## Configuration
- Modify `SCAN_INTERVAL` in `bot.py` for cycle frequency.
- Adjust `SYMBOLS` in `mt5_connector.py` to change traded instruments.

---
**Disclaimer**: Trading involves risk. Use this bot at your own discretion.
