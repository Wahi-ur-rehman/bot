"""
ai_judge.py  ──  AI Signal Confirmation Layer
────────────────────────────────────────────────────
Adds an LLM "second opinion" on top of indicators + ML signals.
Handles API communication with Gemini, Groq, or OpenRouter.
"""

import json
import time
import re
from logger import log_info, log_error

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

AI_PROVIDER   = "gemini"        # "gemini" | "groq" | "openrouter"
AI_MIN_AGREE  = True            # if True, only trade when AI confirms
AI_CACHE_SECS = 300             # don't re-query same symbol within 5 mins
AI_COOLDOWN   = 600             # 10 min cooldown if quota is hit
MAX_RETRIES   = 2               # retry on API failure

# ══════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════

def _build_prompt(symbol: str, signal: dict, df) -> str:
    last = df.iloc[-1]
    candles = df.tail(5)[["open", "high", "low", "close", "volume"]].round(5).to_dict("records")

    return f"""You are a professional forex scalping assistant.

Symbol: {symbol}
Timeframe: M5
Bot decision: {signal['decision']}
Confidence: {signal['confidence']:.1f}%
Market regime: {signal['regime']}
ADX: {signal['adx']:.1f}
Indicator score: {signal['ind_score']:.3f}
Price action score: {signal['pa_score']:.3f}
ML score: {signal['ml_score']:.3f}
EMA trend: {signal['trend']}

Last 5 candles (OHLCV):
{json.dumps(candles, indent=2)}

Current RSI: {last.get('rsi', 'N/A')}
Current MACD diff: {last.get('macd_diff', 'N/A')}
BB position (0=low band, 1=top band): {last.get('bb_position', 'N/A')}
Current ATR: {last.get('atr', 'N/A')}

Based on this data, should the bot execute this {signal['decision']} trade?

Reply ONLY with a JSON object, no extra text:
{{
  "confirmed": true or false,
  "confidence": 0-100,
  "reason": "1-2 sentence explanation",
  "risk_note": "any specific risk to watch (or null)"
}}"""


# ══════════════════════════════════════════════════════════════════
# RESPONSE PARSER
# ══════════════════════════════════════════════════════════════════

def _parse_response(text: str) -> dict:
    try:
        # Strip markdown code fences if present
        clean = re.sub(r"```(?:json)?|```", "", text).strip()
        data = json.loads(clean)
        return {
            "confirmed"  : bool(data.get("confirmed", False)),
            "confidence" : float(data.get("confidence", 50)),
            "reason"     : str(data.get("reason", "no reason")),
            "risk_note"  : data.get("risk_note"),
        }
    except Exception:
        # If parse fails, default to confirming (don't block the bot on API issues)
        log_error(f"[AI] Failed to parse response: {text[:120]}")
        return {"confirmed": True, "confidence": 50,
                "reason": "parse error — defaulting to confirm", "risk_note": None}


# ══════════════════════════════════════════════════════════════════
# PROVIDER IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════

class _GeminiProvider:
    def __init__(self, api_key: str):
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-2.0-flash-lite")
            log_info("[AI] Gemini provider initialized (gemini-2.0-flash-lite).")
        except ImportError:
            raise ImportError("Run: pip install google-generativeai")

    def query(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text


class _GroqProvider:
    def __init__(self, api_key: str):
        try:
            from groq import Groq
            self.client = Groq(api_key=api_key)
            log_info("[AI] Groq provider initialized (llama-3.3-70b-versatile).")
        except ImportError:
            raise ImportError("Run: pip install groq")

    def query(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.1,  # low temp for consistent JSON output
        )
        return response.choices[0].message.content


class _OpenRouterProvider:
    def __init__(self, api_key: str):
        import requests
        self._requests = requests
        self._key = api_key
        self._model = "meta-llama/llama-3.1-8b-instruct:free"
        log_info(f"[AI] OpenRouter provider initialized ({self._model}).")

    def query(self, prompt: str) -> str:
        r = self._requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300,
                "temperature": 0.1,
            },
            timeout=10,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


# ══════════════════════════════════════════════════════════════════
# MAIN CLASS
# ══════════════════════════════════════════════════════════════════

class AIJudge:
    """
    Wraps a free AI provider and provides signal confirmation.
    """

    def __init__(self, provider: str = AI_PROVIDER, api_key: str = ""):
        self._cooldown_until = 0

        if provider == "gemini":
            self._provider = _GeminiProvider(api_key)
        elif provider == "groq":
            self._provider = _GroqProvider(api_key)
        elif provider == "openrouter":
            self._provider = _OpenRouterProvider(api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}. Choose gemini/groq/openrouter")

    def confirm_signal(
        self,
        symbol: str,
        signal: dict,
        df,
        force_refresh: bool = False,
    ) -> dict:
        """
        Ask the AI whether to confirm or veto the given signal.
        """
        # ── Cooldown check ───────────────────────────────────────
        if time.time() < self._cooldown_until:
            return {"confirmed": True, "confidence": 50,
                    "reason": "AI in cooldown (quota) — fallback confirm", "risk_note": None}

        # ── Cache check ──────────────────────────────────────────
        if not force_refresh and symbol in self._cache:
            ts, cached = self._cache[symbol]
            if time.time() - ts < AI_CACHE_SECS:
                return cached

        if signal["decision"] == "HOLD":
            return {"confirmed": False, "confidence": 0,
                    "reason": "HOLD signal — skipped AI check", "risk_note": None}

        prompt = _build_prompt(symbol, signal, df)

        # ── Query with retries ───────────────────────────────────
        for attempt in range(MAX_RETRIES):
            try:
                raw = self._provider.query(prompt)
                verdict = _parse_response(raw)
                self._cache[symbol] = (time.time(), verdict)

                direction = "✅" if verdict["confirmed"] else "❌"
                log_info(
                    f"  🤖 AI {direction} {symbol} {signal['decision']} "
                    f"({verdict['confidence']:.0f}%): {verdict['reason']}"
                )
                if verdict["risk_note"]:
                    log_info(f"     ⚠ Risk: {verdict['risk_note']}")

                return verdict

            except Exception as e:
                # Detect quota error
                if "429" in str(e) or "Quota exceeded" in str(e):
                    log_error(f"[AI] Quota hit for {symbol}. Entering cooldown for {AI_COOLDOWN}s.")
                    self._cooldown_until = time.time() + AI_COOLDOWN
                    break # Exit retry loop early

                log_error(f"[AI] Attempt {attempt+1} failed for {symbol}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2)

        # ── All retries failed — don't block the bot ─────────────
        log_error(f"[AI] All retries failed for {symbol} — defaulting to confirm.")
        return {"confirmed": True, "confidence": 50,
                "reason": "AI unavailable — fallback confirm", "risk_note": None}


def get_news_sentiment(judge: AIJudge, currencies: list[str]) -> dict[str, float]:
    """
    Ask the AI for a sentiment score per currency based on recent context.
    """
    prompt = f"""You are a forex market analyst. Based on your knowledge of recent
macro conditions and typical market sentiment, give a sentiment score for each currency.

Currencies: {', '.join(currencies)}

Reply ONLY with a JSON object, no extra text:
{{
  "EUR": <score from -1.0 to 1.0>,
  "USD": <score from -1.0 to 1.0>,
  ...
  "summary": "1 sentence macro context"
}}"""

    try:
        raw = judge._provider.query(prompt)
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        data = json.loads(clean)
        log_info(f"  🤖 News sentiment: {data.get('summary','')}")
        return {k: float(v) for k, v in data.items() if k != "summary"}
    except Exception as e:
        log_error(f"[AI] News sentiment failed: {e}")
        return {}
