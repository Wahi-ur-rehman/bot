# 🤖 ApexScalp AI v4.0

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Market](https://img.shields.io/badge/market-MT5-green.svg)
![License](https://img.shields.io/badge/license-Proprietary-red.svg)
![AI](https://img.shields.io/badge/AI-Confirmed-gold.svg)

**ApexScalp AI** is a state-of-the-art, multi-symbol scalping bot engineered for MetaTrader 5. It leverages the power of Machine Learning (LightGBM) and Large Language Models (Gemini/Groq) to deliver institutional-grade execution with a 3-layer confirmation logic.

---

## ⚡ Core Pillars

### 🧠 Triple-Confirm Logic
Every trade must pass through a rigorous 3-layer filter:
1.  **Technical Shell**: RSI, MACD, and EMA Trend alignment + ADX Momentum Gate.
2.  **Machine Learning**: LightGBM model trained on hundreds of thousands of data points across 10+ Forex symbols.
3.  **AI Judge**: Real-time signal validation using **Gemini-2.0-Flash** or **Groq (Llama-3.3)**.

### 🛡️ Institutional Risk Management
- **Circuit Breaker**: Stops all trading if daily drawdown exceeds 3%.
- **Adaptive SL/TP**: Dynamically adjusts Stop Loss and Take Profit based on ATR and Market Regime (Trending/Ranging/Volatile).
- **Fractional Kelly Sizing**: Optimizes position sizes based on historical win rates and profit factors.
- **Partial Close**: Automatically secures 50% profit at 1R and moves SL to breakeven.

### 🚀 Performance Engineering
- **Parallel Fetching**: Uses 8-thread concurrent processes to scan symbols 8x faster.
- **Regime Detector**: Automatically identifies market phases to stay out of choppy, low-probability environments.
- **Spread Monitor**: Skips trades if liquidity is thin or spreads are abnormally wide.

---

## 🛠️ Architecture

```mermaid
graph TD
    A[MT5 Market Data] --> B[Analysis Engine]
    B --> C{Indicators + PA}
    B --> D{LightGBM Prediction}
    C --> E[Signal Candidate]
    D --> E
    E --> F{AI Judge Veto?}
    F -- Confirmed --> G[Risk Manager]
    F -- Veto --> H[Log & Skip]
    G --> I[Trade Executor]
    I --> J[Live MT5 Order]
```



## 🔒 License & Security
> [!CAUTION]
> **PROPRIETARY SOFTWARE — ALL RIGHTS RESERVED**
> This code is private property. Unauthorized copying, modification, or redistribution is strictly prohibited. See the [LICENSE](LICENSE) file for full legal terms.

---
**Disclaimer**: *Trading forex and CFDs involves significant risk. This software is provided for educational purposes only. Past performance does not guarantee future results.*
