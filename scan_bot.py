# -*- coding: utf-8 -*-
# Binance USDT Perp Scanner — Railway Cron (runs once; schedule every 2h in Railway)
import ccxt, pandas as pd, numpy as np, time, requests
from datetime import datetime

# Telegram
TELEGRAM_TOKEN = "8331474411:AAG5wW_4m7KuBUdQ8nNi8L1gntx0NN2RUwU"
CHAT_ID = "1982879600"

# Config
TIMEFRAMES = ["1h", "2h", "4h", "1d"]
LOOKBACK_BARS = 400
RECENT_BARS_WINDOW = 4
VOL_MA = 20
EMA_RIBBON = [8, 13, 21, 34, 55]
SMA_FAST = 50
SMA_SLOW = 200
BBANDS_WINDOW = 20
BBANDS_STD = 2.0
SQUEEZE_PERCENTILE = 20
SUCCESS_TARGET_PCT = 0.03
SUCCESS_HORIZON_BARS = 5
MIN_BACKTEST_SIGNALS = 5
REQUEST_SLEEP = 0.12
DEBUG = False
SEND_EMPTY_SUMMARY = True

def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def sma(s, n): return s.rolling(n).mean()

def rsi(series, length=14):
    d = series.diff()
    up = np.where(d > 0, d, 0.0); down = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up, index=series.index).ewm(alpha=1/length, adjust=False).mean()
    rd = pd.Series(down, index=series.index).ewm(alpha=1/length, adjust=False).mean()
    rs = ru / (rd + 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ef, es = ema(series, fast), ema(series, slow)
    m = ef - es; sig = ema(m, signal); h = m - sig
    return m, sig, h

def bollinger_bands(series, window=20, nstd=2.0):
    ma = series.rolling(window).mean(); std = series.rolling(window).std(ddof=0)
    up = ma + nstd*std; lo = ma - nstd*std
    denom = ma.abs().replace(0, np.nan)
    bw = (up - lo) / denom
    return up, ma, lo, bw

def pivot_lows(series, left=2, right=2):
    vals = series.values; n = len(vals); out = []
    for i in range(n):
        if i < left or i > n - right - 1: out.append(False); continue
        win = vals[i-left:i+right+1]; out.append(vals[i] == win.min())
    return pd.Series(out, index=series.index)

def find_bullish_divergence(price, indicator, lookback=80):
    piv_p = pivot_lows(price).tail(lookback)
    piv_i = pivot_lows(indicator).tail(lookback)
    idx = price.tail(lookback).index
    p_lows = [i for i in idx if piv_p.loc[i]]; i_lows = [i for i in idx if piv_i.loc[i]]
    if len(p_lows) < 2 or len(i_lows) < 2: return False
    def nearest(ts, cands): return min(cands, key=lambda x: abs((x - ts).total_seconds()))
    try:
        p1, p2 = p_lows[-2], p_lows[-1]; i1 = nearest(p1, i_lows); i2 = nearest(p2, i_lows)
    except ValueError: return False
    return bool(price.loc[p2] < price.loc[p1] and indicator.loc[i2] > indicator.loc[i1])

def classify_score(score):
    return "LOW" if score == 3 else ("MEDIUM" if score == 4 else ("HIGH" if score >= 5 else "N/A"))

def _fmt_duration(sec):
    sec = int(sec); m, s = divmod(sec, 60); h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s" if h else (f"{m}m {s}s" if m else f"{s}s")

def tg_send(text, disable_web_page_preview=True):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown", "disable_web_page_preview": disable_web_page_preview}
        r = requests.post(url, json=payload, timeout=20)
        if r.status_code != 200: print("[WARN] Telegram send failed:", r.text)
    except Exception as e: print("[WARN] Telegram exception:", e)

def get_exchange():
    return ccxt.binanceusdm({"enableRateLimit": True, "options": {"defaultType": "future", "adjustForTimeDifference": True}})

def fetch_usdt_perp_symbols(ex):
    mk = ex.load_markets(True); syms = []
    for sym, m in mk.items():
        if m.get("contract") and m.get("swap") and m.get("linear") and m.get("quote") == "USDT": syms.append(sym)
    if not syms:
        try:
            fm = ex.fetch_markets()
            for m in fm:
                if m.get("contract") and m.get("swap") and m.get("linear") and m.get("quote") == "USDT":
                    syms.append(m.get("symbol"))
        except Exception as e:
            print("[WARN] fetch_markets fallback failed:", e)
    return sorted(set([s for s in syms if s]))

def fetch_ohlcv_safe(ex, symbol, timeframe, limit):
    for _ in range(3):
        try: return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except ccxt.NetworkError: time.sleep(1.0)
        except Exception as e: print(f"[ERR] fetch_ohlcv {symbol} {timeframe}: {e}"); break
    return None

def compute_indicators(df):
    c = df["close"]; v = df["volume"]
    df["sma_fast"] = sma(c, SMA_FAST); df["sma_slow"] = sma(c, SMA_SLOW)
    df["rsi"] = rsi(c, 14)
    macd_line, signal_line, hist = macd(c)
    df["macd_line"], df["macd_signal"], df["macd_hist"] = macd_line, signal_line, hist
    up, mid, lo, bw = bollinger_bands(c, BBANDS_WINDOW, BBANDS_STD)
    df["bb_upper"], df["bb_mid"], df["bb_lower"], df["bb_bw"] = up, mid, lo, bw
    df["vol_ma"] = sma(v, VOL_MA)
    for L in EMA_RIBBON: df[f"ema_{L}"] = ema(c, L)
    return df

def golden_cross_condition(df):
    f, s = df["sma_fast"], df["sma_slow"]
    return ((f > s) & (f.shift(1) <= s.shift(1))).fillna(False)

def squeeze_condition(df):
    bw = df["bb_bw"]
    perc = bw.rolling(200).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(pd.Series(x).dropna())>0 else np.nan)
    return (perc * 100 <= SQUEEZE_PERCENTILE).fillna(False)

def ema_ribbon_bull(df):
    stack = np.ones(len(df), dtype=bool)
    for i in range(len(EMA_RIBBON)-1):
        stack &= (df[f"ema_{EMA_RIBBON[i]}"] > df[f"ema_{EMA_RIBBON[i+1]}"])
    return pd.Series(stack, index=df.index)

def ema_ribbon_flip_condition(df):
    return (ema_ribbon_bull(df) & (~ema_ribbon_bull(df.shift(1)))).fillna(False)

def breakout_volume_spike(df):
    c, u = df["close"], df["bb_upper"]
    vol, vma = df["volume"], df["vol_ma"]
    cross_up = (c > u) & (c.shift(1) <= u.shift(1))
    spike = vol > (vma * 1.5)
    return (cross_up & spike).fillna(False)

def bullish_divergence_condition(df):
    c = df["close"]
    r = find_bullish_divergence(c, df["rsi"], lookback=80)
    m = find_bullish_divergence(c, df["macd_line"], lookback=80)
    out = pd.Series(False, index=df.index); out.iloc[-1] = bool(r or m); return out

def composite_scores(df):
    res = pd.DataFrame({
        "golden_cross": golden_cross_condition(df),
        "bull_div": bullish_divergence_condition(df),
        "bb_squeeze": squeeze_condition(df),
        "ema_ribbon_flip": ema_ribbon_flip_condition(df),
        "breakout_vol_spike": breakout_volume_spike(df),
    }, index=df.index)
    res["score"] = res.sum(axis=1)
    return res

def backtest_success_rate(df, comp):
    close = df["close"]; idx = comp.index[comp["score"] >= 3]
    succ = 0
    for ts in idx[:-SUCCESS_HORIZON_BARS]:
        i = df.index.get_loc(ts); nxt = close.iloc[i+1:i+1+SUCCESS_HORIZON_BARS]
        if len(nxt) == 0: continue
        if (nxt.max() - close.loc[ts]) / close.loc[ts] >= SUCCESS_TARGET_PCT: succ += 1
    total = max(0, len(idx) - SUCCESS_HORIZON_BARS)
    rate = succ/total if total > 0 else None
    return rate, total

def format_alert(symbol, timeframe, when_ts, comp_row, rate, total):
    score = int(comp_row["score"]); grade = classify_score(score)
    parts = []
    for key in ["golden_cross","bull_div","bb_squeeze","ema_ribbon_flip","breakout_vol_spike"]:
        if comp_row.get(key, False): parts.append(key.replace("_"," ").title())
    checklist = ", ".join(parts) if parts else "-"
    rate_txt = "n/a"
    if rate is not None and total >= MIN_BACKTEST_SIGNALS: rate_txt = f"{rate*100:.1f}% (n={total})"
    elif rate is not None: rate_txt = f"{rate*100:.1f}% (limited n={total})"
    ts_str = when_ts.strftime("%Y-%m-%d %H:%M UTC")
    return (f"🚀 {symbol} · {timeframe.upper()}\n"
            f"Score: {score} -> {grade}\n"
            f"Met: {checklist}\n"
            f"Bar time: {ts_str}\n"
            f"Success rate: {rate_txt}")

def scan_once():
    start = datetime.utcnow()
    ex = get_exchange(); symbols = fetch_usdt_perp_symbols(ex)
    print(f"[INFO] Scanning {len(symbols)} USDT perpetuals")
    alerts = []
    for tf in TIMEFRAMES:
        for sym in symbols:
            try:
                raw = fetch_ohlcv_safe(ex, sym, tf, LOOKBACK_BARS)
                if not raw or len(raw) < max(SMA_SLOW, BBANDS_WINDOW) + 10: time.sleep(REQUEST_SLEEP); continue
                df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True); df.set_index("timestamp", inplace=True)
                df = compute_indicators(df); comp = composite_scores(df); rate, total = backtest_success_rate(df, comp)
                recent = comp.tail(RECENT_BARS_WINDOW)
                for idx, row in recent.iterrows():
                    if row["score"] >= 3:
                        alerts.append(format_alert(sym, tf, idx, row, rate, total)); break
                time.sleep(REQUEST_SLEEP)
            except Exception as e:
                if DEBUG: print(f"[ERR] {sym} {tf}: {e}")
                time.sleep(REQUEST_SLEEP)
    dur = datetime.utcnow() - start; dur_txt = _fmt_duration(dur.total_seconds())
    if alerts:
        header = f'Binance USDT Perp Scanner - {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}\nFound {len(alerts)} signals (>=3 criteria):'
        tg_send(header); [tg_send(m) for m in alerts]
        if SEND_EMPTY_SUMMARY:
            lines = [a.splitlines()[0] + " · " + a.splitlines()[1] for a in alerts[:30]]
            summary = "• " + "\n• ".join(lines); tg_send(summary)
    else:
        tf_list = "/".join([t.upper() for t in TIMEFRAMES])
        msg = f'🕗 Scan complete at {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}\nAnalyzed: {len(symbols)} pairs ({tf_list})\nDuration: {dur_txt}\nNo signals found.'
        tg_send(msg)

if __name__ == "__main__":
    scan_once()  # single run; schedule in Railway Cron
