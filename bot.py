import os
import time
import logging
import threading
import requests
import pandas as pd
from flask import Flask
from datetime import datetime

app = Flask(__name__)

# ───── ENV Settings ─────
CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")
TELEGRAM_BOT_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID      = os.getenv("TELEGRAM_CHAT_ID")

# ───── Strategy Parameters ─────
EMA_LEN         = 9
ATR_LEN         = 14
ATR_SL_MULT     = 1.2
ATR_TP1_MULT    = 1.5
ATR_TP2_MULT    = 2.5
RSI_LEN         = 14
RSI_BUY_LVL     = 30
RSI_SELL_LVL    = 70
PIVOT_LOOKBACK  = 5
SIGNAL_COOLDOWN = 1800
HEARTBEAT_INT   = 7200
CHECK_INT       = 600
MONITOR_INT     = 120
SLEEP_HOURS     = (0, 7)

# ───── Tracking ─────
last_signals   = {}
open_positions = {}
daily_signals  = 0
daily_wins     = 0
daily_losses   = 0

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(message)s")

# ───── Telegram Sender ─────
def send_telegram(msg: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    r = requests.post(url, json=payload)
    if r.status_code != 200:
        logging.error(f"Telegram error: {r.text}")

# ───── Data Fetching ─────
def get_data(tf: str, sym: str) -> pd.DataFrame:
    agg = 5 if tf == "5m" else 15
    res = requests.get(
        "https://min-api.cryptocompare.com/data/v2/histominute",
        params={"fsym": sym[:-4], "tsym": "USDT", "limit": 100, "aggregate": agg, "api_key": CRYPTOCOMPARE_API_KEY},
        timeout=10
    ).json()
    df = pd.DataFrame(res["Data"]["Data"])
    df["timestamp"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("timestamp", inplace=True)
    df[["open","high","low","close","vol"]] = df[["open","high","low","close","volumeto"]]
    return df[["open","high","low","close","vol"]]

# ───── Indicators ─────
def pivot_high(df, lb):
    return df['high'].rolling(window=lb*2+1, center=True) \
             .apply(lambda x: x.iloc[lb] == x.max()).fillna(False)

def pivot_low(df, lb):
    return df['low'].rolling(window=lb*2+1, center=True) \
             .apply(lambda x: x.iloc[lb] == x.min()).fillna(False)

def rsi(series, length):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ───── Cooldown ─────
def check_cooldown(sym, direction, idx):
    key = f"{sym}_{direction}"
    if last_signals.get(key) == idx:
        return False
    last_signals[key] = idx
    return True

# ───── Signal Analysis ─────
def analyze_symbol(sym, tf="15m"):
    global daily_signals
    df = get_data(tf, sym)
    if df is None or len(df) < PIVOT_LOOKBACK*2 + 5:
        return None

    # Indicators
    df["EMA9"] = df["close"].ewm(span=EMA_LEN, adjust=False).mean()
    df["RSI"]  = rsi(df["close"], RSI_LEN)
    df["PH"]   = pivot_high(df, PIVOT_LOOKBACK)
    df["PL"]   = pivot_low(df, PIVOT_LOOKBACK)

    prev = df.iloc[-2]
    last = df.iloc[-1]
    idx  = df.index[-1]
    entry = last["close"]
    rsiVal = last["RSI"]

    direction = None
    ob_type = None
    accuracy = None
    early = False

    # Normal Mode: strict on prev pivot
    if prev["PL"] and entry > last["EMA9"] and rsiVal > RSI_BUY_LVL:
        direction = "Long"
        ob_type = "Bull OB"
        accuracy = "🎯 Accuracy: High"
    elif prev["PH"] and entry < last["EMA9"] and rsiVal < RSI_SELL_LVL:
        direction = "Short"
        ob_type = "Bear OB"
        accuracy = "🎯 Accuracy: High"
    # Smart Mode: near threshold
    elif prev["PL"] and entry > last["EMA9"] and RSI_BUY_LVL - 2 < rsiVal <= RSI_BUY_LVL:
        direction = "Long"
        ob_type = "Bull OB"
        accuracy = "⚠️ Accuracy: Medium"
    elif prev["PH"] and entry < last["EMA9"] and RSI_SELL_LVL <= rsiVal < RSI_SELL_LVL + 2:
        direction = "Short"
        ob_type = "Bear OB"
        accuracy = "⚠️ Accuracy: Medium"
    # Early Signal: prev pivot + EMA
    elif prev["PL"] and entry > last["EMA9"]:
        ob_type = "Bull OB"
        early = True
    elif prev["PH"] and entry < last["EMA9"]:
        ob_type = "Bear OB"
        early = True
    else:
        logging.info(f"{sym}: No OB/EMA/RSI signal")
        return None

    # Cooldown
    if not check_cooldown(sym, direction or "early", idx):
        logging.info(f"{sym}: Cooldown active")
        return None

    # Early Alert
    if early:
        msg = (
            f"🟡 *Early Signal Alert*\n"
            f"*Symbol:* `{sym}`\n"
            f"*Potential:* { '🟢 BUY' if ob_type=='Bull OB' else '🔴 SELL'}\n"
            f"*Price:* `{entry:.6f}` | *RSI:* `{rsiVal:.2f}`\n"
            f"*OB Type:* {ob_type}\n"
            f"🔍 Waiting RSI confirmation..."
        )
        return msg

    # Targets & SL
    atr_val = df['high'].rolling(ATR_LEN).mean().iloc[-1]
    if direction == "Long":
        sl  = entry - atr_val * ATR_SL_MULT
        tp1 = entry + atr_val * ATR_TP1_MULT
        tp2 = entry + atr_val * ATR_TP2_MULT
    else:
        sl  = entry + atr_val * ATR_SL_MULT
        tp1 = entry - atr_val * ATR_TP1_MULT
        tp2 = entry - atr_val * ATR_TP2_MULT

    daily_signals += 1
    stars = "🔥🔥🔥"
    msg = (
        f"🚨 *AI Signal Alert*\n"
        f"*Symbol:* `{sym}`\n"
        f"*Signal:* { '🟢 BUY' if direction=='Long' else '🔴 SELL'}\n"
        f"*Type:* {ob_type}\n"
        f"*Price:* `{entry:.6f}`\n"
        f"*SL:* `{sl:.6f}`  *TP1:* `{tp1:.6f}`  *TP2:* `{tp2:.6f}`\n"
        f"*EMA9:* `{last['EMA9']:.4f}`  |  *RSI:* `{rsiVal:.2f}`\n"
        f"{accuracy}\n"
        f"*Strength:* {stars}"
    )
    open_positions[sym] = {"dir": direction, "sl": sl, "tp1": tp1, "tp2": tp2}
    logging.info(f"{sym}: Signal {direction} @ {entry:.6f}")
    return msg

# ───── Alert Routine ─────
def check_and_alert(sym):
    logging.info(f"🔍 Checking {sym}...")
    msg = analyze_symbol(sym, "15m")
    if msg:
        send_telegram(msg)

# ───── Position Monitoring ─────
def monitor_positions():
    global daily_wins, daily_losses
    while True:
        for sym, pos in list(open_positions.items()):
            df = get_data("15m", sym)
            price = df["close"].iloc[-1]
            d = pos["dir"]
            if (d == "Long" and price >= pos["tp2"]) or (d == "Short" and price <= pos["tp2"]):
                daily_wins += 1
                del open_positions[sym]
            elif (d == "Long" and price <= pos["sl"]) or (d == "Short" and price >= pos["sl"]):
                daily_losses += 1
                del open_positions[sym]
        time.sleep(MONITOR_INT)

# ───── Daily Report ─────
def report_daily():
    total = daily_wins + daily_losses
    wr = round(daily_wins/total*100,1) if total > 0 else 0
    send_telegram(
        f"📊 *Daily Report*\n"
        f"Signals: {daily_signals}\n"
        f"✅ Wins: {daily_wins}\n"
        f"❌ Losses: {daily_losses}\n"
        f"🏆 Winrate: {wr}%"
    )

# ───── Main Monitor ─────
def monitor():
    last_hb = 0
    symbols = [
        "BTCUSDT","ETHUSDT","DOGEUSDT","BNBUSDT","XRPUSDT",
        "RENDERUSDT","TRUMPUSDT","FARTCOINUSDT","XLMUSDT",
        "SHIBUSDT","ADAUSDT","NOTUSDT","PROMUSPTUSDT","PENDLEUSDT"
    ]
    while True:
        now = datetime.utcnow()
        hr = (now.hour + 3) % 24
        mn = now.minute
        if SLEEP_HOURS[0] <= hr < SLEEP_HOURS[1]:
            time.sleep(60)
            continue
        if time.time() - last_hb > HEARTBEAT_INT:
            send_telegram("🤖 Bot live and scanning.")
            last_hb = time.time()
        threads = []
        for s in symbols:
            t = threading.Thread(target=check_and_alert, args=(s,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        if hr == 23 and mn >= 55:
            report_daily()
        time.sleep(CHECK_INT)

@app.route("/")
def home():
    return "✅ Crypto Signal Bot is running."

if __name__ == "__main__":
    threading.Thread(target=monitor, daemon=True).start()
    threading.Thread(target=monitor_positions, daemon=True).start()
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
