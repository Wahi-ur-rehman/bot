# 📊 ApexScalp AI v3.0: Technical Proposal

## 1. Executive Summary
**ApexScalp AI** is a multi-layer, autonomous trading system designed for institutional-grade precision. It utilizes a hybrid approach, combining traditional Technical Analysis (RSI, MACD, EMA), Price Action (Patterns), and Ensemble Machine Learning (Random Forest) to execute high-frequency scalping trades on **MetaTrader 5**.

---

## 2. Library Ecosystem & Rationale
The bot is built on a modern Python stack for low latency and high reliability:
- **`MetaTrader5`**: Native bridge for lightning-fast execution.
- **`LightGBM`**: High-performance gradient boosting for price movement prediction.
- **`Pandas/NumPy`**: Vectorized indicator calculations.
- **`Google Generative AI`**: LLM-based signal confirmation (Gemini integration).

---

## 3. Functional Blueprint (Core Operations)

| Component | Responsibility | Key Function |
| :--- | :--- | :--- |
| **MT5 Connector** | Data pipeline & Execution | `fetch_all_symbols()` |
| **Analysis Engine** | Indicator & ML logic | `combined_signal()` |
| **Risk Manager** | SL/TP & Sizing | `calculate_sl_tp()` |
| **AI Judge** | LLM Validation | `confirm_signal()` |
| **Trade Executor** | Order routing | `execute_signals()` |

---

## 4. The Confluence Logic (Weighted Consensus)
The bot uses a 'weighted consensus' model to filter market noise and ensure high-probability entries:

1.  **Technical Indicators (35% Weight)**: RSI oversold/overbought, MACD crossovers, and EMA trend alignment.
2.  **Price Action Patterns (25% Weight)**: Real-time detection of Hammer, Engulfing, and Morning/Evening Star patterns.
3.  **Machine Learning (40% Weight)**: Directional probability scores from trained LightGBM models.

---

## 5. Risk Protocol
- **Risk Per Trade**: 0.5% Fixed Fractional.
- **Stop Loss Filter**: 1.5x ATR (Dynamic Volatility adjustment).
- **Reward Ratio**: 1:2 Minimum Target.
- **Max Capacity**: 8 Simultaneous Positions.
- **Circuit Breaker**: Automatic halt at 3% daily drawdown.

---

> [!IMPORTANT]
> This proposal outlines the technical architecture of the ApexScalp AI system. Unauthorized reproduction or use is strictly prohibited.
