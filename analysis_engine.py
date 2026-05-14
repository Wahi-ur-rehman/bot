"""
analysis_engine.py  ── v4 (Production)
────────────────────────────────────────
Four analysis layers → combined score [-1, +1]

  1. Technical indicators  – RSI, MACD, EMA trend, ADX, ATR, BB
  2. Price action          – Candlestick patterns with trend context
  3. ML prediction         – LightGBM (replaces RandomForest; 10x faster,
                             better accuracy, handles imbalanced classes)
  4. Signal persistence    – N consecutive agreeing cycles to confirm

v4 upgrades over v3:
  ✔ LightGBM instead of RandomForestClassifier
  ✔ 17-feature set (was 10): adds body_ratio, wick_ratio, momentum,
    vol_trend, atr, atr_rank, efficiency_ratio, bb_position
  ✔ Bollinger Bands added to compute_indicators()
  ✔ ATR column computed in indicators (needed by risk_manager)
  ✔ MarketRegimeDetector class (ADX + efficiency ratio + ATR rank)
  ✔ train_multi() for multi-symbol combined training
  ✔ shuffle=False + class_weight preserved (no lookahead, no bias)
  ✔ Graceful LightGBM fallback to RandomForest if lgbm not installed
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from ta.momentum  import RSIIndicator
from ta.trend     import MACD, ADXIndicator, EMAIndicator
from ta.volatility import BollingerBands
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler

# ── LightGBM with graceful fallback ─────────────────────────────────
try:
    from lightgbm import LGBMClassifier
    _USE_LGBM = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    _USE_LGBM = False
    print("[WARNING] lightgbm not found — falling back to RandomForest. "
          "Run: pip install lightgbm")


# ════════════════════════════════════════════════════════════════════════════
#  1. INDICATORS
# ════════════════════════════════════════════════════════════════════════════

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators needed for signals and ML features.
    Returns a copy with NaN rows dropped.
    """
    df = df.copy()

    # ── RSI ─────────────────────────────────────────────────────────
    df["rsi"]      = RSIIndicator(close=df["close"], window=14).rsi()

    # ── MACD ────────────────────────────────────────────────────────
    macd_obj          = MACD(close=df["close"])
    df["macd"]        = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_diff"]   = macd_obj.macd_diff()

    # ── EMAs ────────────────────────────────────────────────────────
    df["ema20"]  = df["close"].ewm(span=20,  adjust=False).mean()
    df["ema50"]  = df["close"].ewm(span=50,  adjust=False).mean()
    df["ema100"] = df["close"].ewm(span=100, adjust=False).mean()

    # ── ADX ─────────────────────────────────────────────────────────
    adx_obj       = ADXIndicator(high=df["high"], low=df["low"],
                                  close=df["close"], window=14)
    df["adx"]     = adx_obj.adx()
    df["adx_pos"] = adx_obj.adx_pos()
    df["adx_neg"] = adx_obj.adx_neg()

    # ── ATR (True Range EMA) ─────────────────────────────────────────
    high  = df["high"]
    low   = df["low"]
    prev  = df["close"].shift(1)
    tr    = pd.concat([high - low,
                        (high - prev).abs(),
                        (low  - prev).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=14, adjust=False).mean()

    # ATR rank (percentile over last 100 bars) — proxy for vol regime
    df["atr_rank"] = df["atr"].rolling(100).rank(pct=True)

    # ── Bollinger Bands ──────────────────────────────────────────────
    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_upper"]    = bb.bollinger_hband()
    df["bb_lower"]    = bb.bollinger_lband()
    df["bb_mid"]      = bb.bollinger_mavg()
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (
        (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    )

    # ── Volume ──────────────────────────────────────────────────────
    df["returns"]   = df["close"].pct_change()
    df["range"]     = df["high"] - df["low"]
    df["vol_ratio"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-9)
    df["vol_trend"] = df["volume"].rolling(5).mean() / (
        df["volume"].rolling(20).mean() + 1e-9
    )

    # ── Price microstructure features ───────────────────────────────
    body          = df["close"] - df["open"]
    full_range    = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"]     = body.abs() / full_range
    df["wick_ratio"]     = (full_range - body.abs()) / full_range
    df["price_momentum"] = df["close"].pct_change(5)   # 5-bar momentum

    # ── Hurst-proxy: efficiency ratio (20-bar) ──────────────────────
    change       = (df["close"] - df["close"].shift(20)).abs()
    path         = df["close"].diff().abs().rolling(20).sum()
    df["efficiency_ratio"] = change / (path + 1e-9)

    return df.dropna()


# ════════════════════════════════════════════════════════════════════════════
#  MARKET REGIME DETECTOR
# ════════════════════════════════════════════════════════════════════════════

class MarketRegimeDetector:
    """
    Classifies current market into one of four regimes:

      TRENDING  — strong directional move (best for scalping)
      RANGING   — sideways / mean-reverting (skip new entries)
      VOLATILE  — high ATR percentile (reduce size or skip)
      NEUTRAL   — ambiguous (apply normal filters)

    Based on:
      - ADX strength (>25 = trending, <20 = ranging)
      - Efficiency ratio (Hurst proxy): >0.35 → directional
      - ATR rank: >0.85 → volatile outlier session
    """

    @staticmethod
    def detect(df: pd.DataFrame) -> str:
        if len(df) < 50:
            return "NEUTRAL"

        last = df.iloc[-1]

        adx      = float(last.get("adx", 0))
        eff      = float(last.get("efficiency_ratio", 0.3))
        atr_rank = float(last.get("atr_rank", 0.5))

        if atr_rank > 0.85:
            return "VOLATILE"      # unusual volatility spike — be cautious
        elif adx > 25 and eff > 0.35:
            return "TRENDING"      # strong trend — go with it
        elif adx < 20 and eff < 0.25:
            return "RANGING"       # sideways chop — avoid
        else:
            return "NEUTRAL"       # moderate / ambiguous


# ════════════════════════════════════════════════════════════════════════════
#  INDICATOR SIGNAL
# ════════════════════════════════════════════════════════════════════════════

def indicator_signal(df: pd.DataFrame) -> float:
    """Returns a score in [-1, +1] based on RSI, MACD, EMA alignment."""
    if len(df) < 3:
        return 0.0

    last = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0.0

    bullish_trend = (last["ema20"] > last["ema50"] > last["ema100"])
    bearish_trend = (last["ema20"] < last["ema50"] < last["ema100"])

    # RSI
    rsi = last["rsi"]
    if rsi < 30:
        score += 1.0 if bullish_trend or not bearish_trend else 0.3
    elif rsi < 40:
        score += 0.5 if not bearish_trend else 0.1
    elif rsi > 70:
        score -= 1.0 if bearish_trend or not bullish_trend else 0.3
    elif rsi > 60:
        score -= 0.5 if not bullish_trend else 0.1

    # MACD crossover
    prev_diff = prev["macd_diff"]
    curr_diff = last["macd_diff"]
    if prev_diff < 0 and curr_diff > 0:
        score += 1.0 if not bearish_trend else 0.2
    elif prev_diff > 0 and curr_diff < 0:
        score -= 1.0 if not bullish_trend else 0.2
    elif curr_diff > 0:
        score += 0.3
    else:
        score -= 0.3

    # EMA alignment
    if bullish_trend:
        score += 0.4
    elif bearish_trend:
        score -= 0.4

    # Bollinger position (price near upper/lower band)
    if "bb_position" in df.columns:
        bp = float(last.get("bb_position", 0.5))
        if bp < 0.1:
            score += 0.3  # near lower band — potential bounce
        elif bp > 0.9:
            score -= 0.3  # near upper band — potential rejection

    # ADX regime dampener
    adx = last["adx"]
    if adx < 12:
        score *= 0.4

    return max(-1.0, min(1.0, score / 2.7))


# ════════════════════════════════════════════════════════════════════════════
#  PRICE ACTION SIGNAL
# ════════════════════════════════════════════════════════════════════════════

def price_action_signal(df: pd.DataFrame) -> float:
    """Returns a score in [-1, +1] from candlestick pattern recognition."""
    if len(df) < 3:
        return 0.0

    c0 = df.iloc[-1]
    c1 = df.iloc[-2]
    c2 = df.iloc[-3]

    body0 = abs(c0["close"] - c0["open"])
    body1 = abs(c1["close"] - c1["open"])
    score = 0.0

    bullish_trend = bearish_trend = False
    if "ema20" in df.columns and "ema50" in df.columns:
        bullish_trend = float(c0["ema20"]) > float(c0["ema50"])
        bearish_trend = float(c0["ema20"]) < float(c0["ema50"])

    lower_wick = min(c0["open"], c0["close"]) - c0["low"]
    upper_wick = c0["high"] - max(c0["open"], c0["close"])

    # Hammer
    if body0 > 0 and lower_wick > 2 * body0 and upper_wick < body0 * 0.5:
        score += 0.8 if bullish_trend else 0.3

    # Bullish engulfing
    if (c1["close"] < c1["open"] and c0["close"] > c0["open"] and
            c0["open"] < c1["close"] and c0["close"] > c1["open"]):
        score += 1.0 if bullish_trend else 0.4

    # Morning star
    if (c2["close"] < c2["open"] and body1 < body0 * 0.3 and
            c0["close"] > c0["open"] and
            c0["close"] > (c2["open"] + c2["close"]) / 2):
        score += 0.9 if bullish_trend else 0.3

    # Shooting star
    if body0 > 0 and upper_wick > 2 * body0 and lower_wick < body0 * 0.5:
        score -= 0.8 if bearish_trend else 0.3

    # Bearish engulfing
    if (c1["close"] > c1["open"] and c0["close"] < c0["open"] and
            c0["open"] > c1["close"] and c0["close"] < c1["open"]):
        score -= 1.0 if bearish_trend else 0.4

    # Evening star
    if (c2["close"] > c2["open"] and body1 < body0 * 0.3 and
            c0["close"] < c0["open"] and
            c0["close"] < (c2["open"] + c2["close"]) / 2):
        score -= 0.9 if bearish_trend else 0.3

    # Triple EMA bias
    if "ema100" in df.columns:
        if float(c0["ema20"]) > float(c0["ema50"]) > float(c0["ema100"]):
            score += 0.2
        elif float(c0["ema20"]) < float(c0["ema50"]) < float(c0["ema100"]):
            score -= 0.2

    return max(-1.0, min(1.0, score))


# ════════════════════════════════════════════════════════════════════════════
#  ML MODEL  (LightGBM / RandomForest fallback)
# ════════════════════════════════════════════════════════════════════════════

class MLSignal:
    """
    Predicts trade direction using gradient boosting (LightGBM preferred).

    Features (17):  rsi, macd_diff, returns, range, ema20, ema50, adx,
                    adx_pos, adx_neg, vol_ratio, body_ratio, wick_ratio,
                    price_momentum, vol_trend, atr_rank, efficiency_ratio,
                    bb_position
    Label:           -1 (SELL) | 0 (HOLD) | 1 (BUY)  over 5-bar horizon
    """

    FEATURES = [
        "rsi", "macd_diff", "returns", "range",
        "ema20", "ema50", "adx", "adx_pos", "adx_neg", "vol_ratio",
        "body_ratio", "wick_ratio", "price_momentum", "vol_trend",
        "atr_rank", "efficiency_ratio", "bb_position",
    ]

    def __init__(self):
        if _USE_LGBM:
            self.model = LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                class_weight="balanced",
                random_state=42,
                verbose=-1,
            )
        else:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
            )
        self.scaler  = StandardScaler()
        self.trained = False
        self._model_name = "LightGBM" if _USE_LGBM else "RandomForest"

    def _build_labels(self, df: pd.DataFrame, lookahead: int = 5) -> pd.Series:
        future = df["close"].shift(-lookahead)
        change = (future - df["close"]) / df["close"]
        labels = pd.cut(
            change,
            bins=[-np.inf, -0.001, 0.001, np.inf],
            labels=[-1, 0, 1],
        ).astype(float)
        return labels

    def train(self, df: pd.DataFrame) -> bool:
        """Train on a single symbol's indicator DataFrame."""
        if len(df) < 100:
            print("[ML] Need 100+ candles.")
            return False

        df = compute_indicators(df.copy())
        available = [f for f in self.FEATURES if f in df.columns]
        if len(available) < 10:
            print(f"[ML] Too few features available ({len(available)}).")
            return False

        labels        = self._build_labels(df)
        X             = df[available].copy()
        y             = labels
        mask          = y.notna() & X.notna().all(axis=1)
        X, y          = X[mask], y[mask]

        if len(X) < 60:
            return False

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        X_tr_s = self.scaler.fit_transform(X_tr)
        X_te_s = self.scaler.transform(X_te)

        self.model.fit(X_tr_s, y_tr)
        acc = self.model.score(X_te_s, y_te)
        self.trained = True
        print(f"[ML] {self._model_name} trained — acc={acc:.1%}  rows={len(X_tr)}")
        return True

    def train_multi(self, frames: dict) -> bool:
        """
        Train on combined data from multiple symbols for better generalization.
        `frames` is {symbol: raw_ohlcv_df}.
        """
        combined_parts = []
        for sym, df in frames.items():
            if len(df) < 100:
                continue
            df_ind = compute_indicators(df.copy())
            combined_parts.append(df_ind)

        if not combined_parts:
            return False

        full_df = pd.concat(combined_parts, ignore_index=False)
        return self.train(full_df)

    def predict(self, df: pd.DataFrame) -> float:
        """Returns score in [-1, +1]: positive = bullish, negative = bearish."""
        if not self.trained or len(df) < 20:
            return 0.0

        available = [f for f in self.FEATURES if f in df.columns]
        if len(available) < 10:
            return 0.0

        last_row = df[available].iloc[[-1]]
        if last_row.isnull().any().any():
            return 0.0

        # Scaler was fit on potentially fewer features — align columns
        try:
            last_scaled = self.scaler.transform(last_row)
        except Exception:
            return 0.0

        proba   = self.model.predict_proba(last_scaled)[0]
        classes = list(self.model.classes_)

        p_sell = proba[classes.index(-1.0)] if -1.0 in classes else 0.0
        p_buy  = proba[classes.index( 1.0)] if  1.0 in classes else 0.0

        return round(float(p_buy - p_sell), 4)


# ════════════════════════════════════════════════════════════════════════════
#  SIGNAL PERSISTENCE
# ════════════════════════════════════════════════════════════════════════════

class SignalPersistence:
    """
    Requires the same non-HOLD decision N consecutive cycles.
    Call reset_all() after all positions close to allow fresh entries.
    """

    def __init__(self, required_streak: int = 2):
        self.required            = required_streak
        self._streaks: dict[str, int] = {}
        self._last:    dict[str, str] = {}

    def confirm(self, symbol: str, decision: str) -> bool:
        if decision == "HOLD":
            self._streaks[symbol] = 0
            self._last[symbol]    = "HOLD"
            return False

        prev = self._last.get(symbol, "HOLD")
        if decision == prev:
            self._streaks[symbol] = self._streaks.get(symbol, 0) + 1
        else:
            self._streaks[symbol] = 1

        self._last[symbol] = decision

        if self._streaks[symbol] >= self.required:
            self._streaks[symbol] = 0
            return True

        return False

    def reset_all(self):
        self._streaks.clear()
        self._last.clear()


# ════════════════════════════════════════════════════════════════════════════
#  COMBINED SIGNAL
# ════════════════════════════════════════════════════════════════════════════

ADX_MIN               = 12.0   # minimum trend strength to consider
DECISION_THRESHOLD    = 0.20   # score threshold for BUY/SELL
MIN_SIGNAL_CONFIDENCE = 15.0   # pre-filter; bot.py MIN_CONFIDENCE is the gate


def combined_signal(
    df       : pd.DataFrame,
    ml_model : MLSignal,
    weights  : tuple = (0.35, 0.25, 0.40),
) -> dict:
    """
    Combine indicator + price-action + ML scores into a single decision.

    Returns:
        dict with keys: score, decision, confidence, adx, trend,
                        ind_score, pa_score, ml_score, regime
    """
    df_ind = compute_indicators(df)

    if df_ind.empty:
        return _hold_result(0, 0, 0, 0, "no_data")

    last = df_ind.iloc[-1]
    adx  = float(last.get("adx", 0))

    if adx < ADX_MIN:
        return _hold_result(0, 0, 0, adx, "sideways")

    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    trend = "BULL" if ema20 > ema50 else ("BEAR" if ema20 < ema50 else "FLAT")

    # Compute regime for reporting (bot.py uses this for filtering)
    regime = MarketRegimeDetector.detect(df_ind)

    ind_score = indicator_signal(df_ind)
    pa_score  = price_action_signal(df_ind)
    ml_score  = ml_model.predict(df_ind)

    w_ind, w_pa, w_ml = weights
    score = (w_ind * ind_score) + (w_pa * pa_score) + (w_ml * ml_score)
    score = max(-1.0, min(1.0, score))

    # Trend conflict dampener
    if score > 0 and trend == "BEAR":
        score *= 0.7
    elif score < 0 and trend == "BULL":
        score *= 0.7

    if score >= DECISION_THRESHOLD:
        decision = "BUY"
    elif score <= -DECISION_THRESHOLD:
        decision = "SELL"
    else:
        decision = "HOLD"

    confidence = abs(score) * 100

    if confidence < MIN_SIGNAL_CONFIDENCE:
        decision = "HOLD"

    return {
        "score"      : round(score, 4),
        "decision"   : decision,
        "ind_score"  : round(ind_score, 4),
        "pa_score"   : round(pa_score,  4),
        "ml_score"   : round(ml_score,  4),
        "confidence" : round(confidence, 1),
        "adx"        : round(adx, 1),
        "trend"      : trend,
        "regime"     : regime,
    }


def _hold_result(ind, pa, ml, adx, trend):
    return {
        "score": 0.0, "decision": "HOLD",
        "ind_score": float(ind), "pa_score": float(pa), "ml_score": float(ml),
        "confidence": 0.0, "adx": float(adx), "trend": str(trend),
        "regime": "NEUTRAL",
    }
