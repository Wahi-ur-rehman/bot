"""
analysis_engine.py  ── v3 (FIXED)
──────────────────────────────────
Four analysis layers → combined score [-1, +1]

  1. Technical indicators  – RSI, MACD, EMA trend, ADX regime
  2. Price action          – candlestick patterns with trend context
  3. ML prediction         – RandomForest (NaN-safe, no lookahead leak)
  4. Signal persistence    – N consecutive agreeing cycles to confirm

v3 fixes
  ✔ ADX_MIN lowered 15 → 12  (M15 forex rarely hits 20+; was blocking all trades)
  ✔ DECISION_THRESHOLD lowered 0.25 → 0.22  (works with confidence gate, not against it)
  ✔ MIN_SIGNAL_CONFIDENCE set to 20.0 (gate is in bot.py at 25, not doubled here)
  ✔ _build_labels: astype(float) — NaN-safe crash fix from v2 kept
  ✔ shuffle=False in train_test_split — no lookahead data leak
  ✔ class_weight="balanced" — no HOLD bias in ML
"""

import numpy as np
import pandas as pd
from ta.momentum  import RSIIndicator
from ta.trend     import MACD, ADXIndicator
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════════════════
#  1. INDICATORS
# ════════════════════════════════════════════════════════════════════════════

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rsi_obj       = RSIIndicator(close=df["close"], window=14)
    df["rsi"]     = rsi_obj.rsi()

    macd_obj          = MACD(close=df["close"])
    df["macd"]        = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_diff"]   = macd_obj.macd_diff()

    df["ema20"]  = df["close"].ewm(span=20).mean()
    df["ema50"]  = df["close"].ewm(span=50).mean()
    df["ema100"] = df["close"].ewm(span=100).mean()

    adx_obj       = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["adx"]     = adx_obj.adx()
    df["adx_pos"] = adx_obj.adx_pos()
    df["adx_neg"] = adx_obj.adx_neg()

    df["returns"]   = df["close"].pct_change()
    df["range"]     = df["high"] - df["low"]
    df["vol_ratio"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-9)

    return df.dropna()


def indicator_signal(df: pd.DataFrame) -> float:
    if len(df) < 3:
        return 0.0

    last = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0.0

    bullish_trend = (last["ema20"] > last["ema50"] > last["ema100"])
    bearish_trend = (last["ema20"] < last["ema50"] < last["ema100"])

    rsi = last["rsi"]
    if rsi < 30:
        score += 1.0 if bullish_trend or not bearish_trend else 0.3
    elif rsi < 40:
        score += 0.5 if not bearish_trend else 0.1
    elif rsi > 70:
        score -= 1.0 if bearish_trend or not bullish_trend else 0.3
    elif rsi > 60:
        score -= 0.5 if not bullish_trend else 0.1

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

    if bullish_trend:
        score += 0.4
    elif bearish_trend:
        score -= 0.4

    adx = last["adx"]
    if adx < 12:
        score *= 0.4

    return max(-1.0, min(1.0, score / 2.7))


# ════════════════════════════════════════════════════════════════════════════
#  2. PRICE ACTION
# ════════════════════════════════════════════════════════════════════════════

def price_action_signal(df: pd.DataFrame) -> float:
    if len(df) < 3:
        return 0.0

    c0 = df.iloc[-1]
    c1 = df.iloc[-2]
    c2 = df.iloc[-3]

    body0 = abs(c0["close"] - c0["open"])
    body1 = abs(c1["close"] - c1["open"])
    score = 0.0

    bullish_trend = False
    bearish_trend = False
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

    if "ema100" in df.columns:
        if float(c0["ema20"]) > float(c0["ema50"]) > float(c0["ema100"]):
            score += 0.2
        elif float(c0["ema20"]) < float(c0["ema50"]) < float(c0["ema100"]):
            score -= 0.2

    return max(-1.0, min(1.0, score))


# ════════════════════════════════════════════════════════════════════════════
#  3. ML MODEL
# ════════════════════════════════════════════════════════════════════════════

class MLSignal:
    FEATURES = [
        "rsi", "macd_diff", "returns", "range",
        "ema20", "ema50", "adx", "adx_pos", "adx_neg", "vol_ratio"
    ]

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42
        )
        self.scaler  = StandardScaler()
        self.trained = False

    def _build_labels(self, df: pd.DataFrame, lookahead: int = 5) -> pd.Series:
        future = df["close"].shift(-lookahead)
        change = (future - df["close"]) / df["close"]
        labels = pd.cut(
            change,
            bins=[-np.inf, -0.001, 0.001, np.inf],
            labels=[-1, 0, 1]
        ).astype(float)   # NaN-safe fix
        return labels

    def train(self, df: pd.DataFrame) -> bool:
        if len(df) < 100:
            print("[ML] Need 100+ candles.")
            return False

        df = compute_indicators(df.copy())
        missing = [f for f in self.FEATURES if f not in df.columns]
        if missing:
            print(f"[ML] Missing: {missing}")
            return False

        labels = self._build_labels(df)
        X = df[self.FEATURES].copy()
        y = labels
        mask = y.notna()
        X, y = X[mask], y[mask]

        if len(X) < 60:
            return False

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_tr_s = self.scaler.fit_transform(X_tr)
        X_te_s = self.scaler.transform(X_te)

        self.model.fit(X_tr_s, y_tr)
        acc = self.model.score(X_te_s, y_te)
        self.trained = True
        print(f"[ML] Trained — accuracy: {acc:.1%}  rows={len(X_tr)}")
        return True

    def predict(self, df: pd.DataFrame) -> float:
        if not self.trained or len(df) < 20:
            return 0.0

        missing = [f for f in self.FEATURES if f not in df.columns]
        if missing:
            return 0.0

        last_row = df[self.FEATURES].iloc[[-1]]
        if last_row.isnull().any().any():
            return 0.0

        last_scaled = self.scaler.transform(last_row)
        proba       = self.model.predict_proba(last_scaled)[0]
        classes     = list(self.model.classes_)

        p_sell = proba[classes.index(-1.0)] if -1.0 in classes else 0.0
        p_buy  = proba[classes.index( 1.0)] if  1.0 in classes else 0.0

        return round(float(p_buy - p_sell), 4)


# ════════════════════════════════════════════════════════════════════════════
#  4. SIGNAL PERSISTENCE
# ════════════════════════════════════════════════════════════════════════════

class SignalPersistence:
    """
    Requires same non-HOLD decision N cycles in a row.
    Call reset_all() when all positions are manually closed so
    the bot re-evaluates from a clean state.
    """

    def __init__(self, required_streak: int = 2):
        self.required  = required_streak
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
        """Call this after all positions are closed to allow fresh entries."""
        self._streaks.clear()
        self._last.clear()


# ════════════════════════════════════════════════════════════════════════════
#  COMBINED SIGNAL
# ════════════════════════════════════════════════════════════════════════════

ADX_MIN               = 10.0   # M15 forex ranging market: 10 is the real floor
DECISION_THRESHOLD    = 0.18   # lower so moderate confluence signals pass through
MIN_SIGNAL_CONFIDENCE = 15.0   # pre-filter; bot.py MIN_CONFIDENCE is the real gate


def combined_signal(
    df       : pd.DataFrame,
    ml_model : MLSignal,
    weights  : tuple = (0.35, 0.25, 0.40)
) -> dict:
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

    ind_score = indicator_signal(df_ind)
    pa_score  = price_action_signal(df_ind)
    ml_score  = ml_model.predict(df_ind)

    w_ind, w_pa, w_ml = weights
    score = (w_ind * ind_score) + (w_pa * pa_score) + (w_ml * ml_score)
    score = max(-1.0, min(1.0, score))

    # Trend conflict dampener (0.7 = moderate penalty, not a full kill)
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
    }


def _hold_result(ind, pa, ml, adx, trend):
    return {
        "score": 0.0, "decision": "HOLD",
        "ind_score": float(ind), "pa_score": float(pa), "ml_score": float(ml),
        "confidence": 0.0, "adx": float(adx), "trend": str(trend),
    }
