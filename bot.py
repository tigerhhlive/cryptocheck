import os
import time
import logging
import threading
import requests
import pandas as pd
import pandas_ta as ta
from flask import Flask
from datetime import datetime

app = Flask(__name__)

# --- Environment Variables ---
CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")
TELEGRAM_BOT_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID       = os.getenv("TELEGRAM_CHAT_ID")

# --- Strategy Settings ---
EMA_LEN        = 9
ATR_LEN        = 14
ATR_SL_MULT    = 1.2
ATR_TP1_MULT   = 1.5
ATR_TP2_MULT   = 2.5
RSI_LEN        = 14
RSI_BUY        = 30
RSI_SELL       = 70
PIVOT_LOOKBACK = 5
SIGNAL_COOLDOWN = 1800
HEARTBEAT_INT   = 7200
CHECK_INT       = 600
MONITOR_INT     = 120
SLEEP_HOURS     = (0, 7)

# --- Runtime State ---
last_signals   = {}
open_positions = {}
daily_signals  = 0
daily_wins     = 0
daily_losses   = 0

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(message)s")

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload)
        if r.status_code != 200:
            logging.error(f"Telegram error: {r.text}")
    except Exception as e:
        logging.error(f"Telegram send failed: {e}")

def get_data(tf, symbol):
    agg = 5 if tf == "5m" else 15
    url = "https://min-api.cryptocompare.com/data/v2/histominute"
    params = {
        "fsym": symbol[:-4],
        "tsym": "USDT",
        "limit": 100,
        "aggregate": agg,
        "api_key": CRYPTOCOMPARE_API_KEY
    }
    r = requests.get(url, params=params)
    data = r.json().get("Data", {}).get("Data", [])
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("timestamp", inplace=True)
    df[["open", "high", "low", "close", "vol"]] = df[["open", "high", "low", "close", "volumeto"]]
    return df[["open", "high", "low", "close", "vol"]]

def pivot_high(df, lb):
    ph = df['high'].rolling(window=lb*2+1, center=True).apply(lambda x: 1 if x.iloc[lb]==x.max() else 0)
    return ph == 1

def pivot_low(df, lb):
    pl = df['low'].rolling(window=lb*2+1, center=True).apply(lambda x: 1 if x.iloc[lb]==x.min() else 0)
    return pl == 1

def check_cooldown(symbol, direction, idx):
    key = f"{symbol}_{direction}"
    if last_signals.get(key) == idx:
        return False
    last_signals[key] = idx
    return True

def analyze_symbol(symbol, tf="15m"):
    global daily_signals
    df = get_data(tf, symbol)
    if len(df) < PIVOT_LOOKBACK*2+5:
        return None

    df["EMA"] = ta.ema(df["close"], length=EMA_LEN)
    df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=ATR_LEN)
    df["RSI"] = ta.rsi(df["close"], length=RSI_LEN)
    df["PH"]  = pivot_high(df, PIVOT_LOOKBACK)
    df["PL"]  = pivot_low(df, PIVOT_LOOKBACK)

    last = df.iloc[-1]
    prev = df.iloc[-2]
    idx  = df.index[-1]

    direction = None
    entry     = last["close"]
    atr       = last["ATR"]
    rsi_val   = last["RSI"]

    if prev["PL"] and last["close"] > last["EMA"] and rsi_val > RSI_BUY:
        direction = "Long"
    elif prev["PH"] and last["close"] < last["EMA"] and rsi_val < RSI_SELL:
        direction = "Short"

    if not direction:
        logging.info(f"{symbol}: No valid signal condition met.")
        return None

    if not check_cooldown(symbol, direction, idx):
        logging.info(f"{symbol}: Signal cooldown active.")
        return None

    # Calculate TP/SL
    if direction == "Long":
        sl  = entry - atr * ATR_SL_MULT
        tp1 = entry + atr * ATR_TP1_MULT
        tp2 = entry + atr * ATR_TP2_MULT
    else:
        sl  = entry + atr * ATR_SL_MULT
        tp1 = entry - atr * ATR_TP1_MULT
        tp2 = entry - atr * ATR_TP2_MULT

    open_positions[symbol] = {"dir": direction, "sl": sl, "tp1": tp1, "tp2": tp2}
    daily_signals += 1
    logging.info(f"{symbol}: Signal {direction} @ {entry:.6f}")

    message = (
        f"🚨 *AI Signal Alert*\n"
        f"*Symbol:* `{symbol}`\n"
        f"*Signal:* {'🟢 BUY' if direction == 'Long' else '🔴 SELL'}\n"
        f"*Price:* `{entry:.6f}`\n"
        f"*SL:* `{sl:.6f}`  *TP1:* `{tp1:.6f}`  *TP2:* `{tp2:.6f}`\n"
        f"*EMA9:* `{last['EMA']:.4f}` | *RSI:* `{rsi_val:.2f}`"
    )

    return message

def analyze_symbol_mtf(symbol):
    m5  = analyze_symbol(symbol, "5m")
    m15 = analyze_symbol(symbol, "15m")
    if m5 and m15 and (("BUY" in m5 and "BUY" in m15) or ("SELL" in m5 and "SELL" in m15)):
        return m15
    return None

def check_and_alert(symbol):
    logging.info(f"🔍 Checking {symbol}...")
    msg = analyze_symbol_mtf(symbol)
    if msg:
        send_telegram(msg)

def monitor_positions():
    global daily_wins, daily_losses
    while True:
        for symbol, pos in list(open_positions.items()):
            df = get_data("15m", symbol)
            price = df["close"].iloc[-1]
            d = pos["dir"]
            if (d == "Long" and price >= pos["tp2"]) or (d == "Short" and price <= pos["tp2"]):
                daily_wins += 1
                del open_positions[symbol]
            elif (d == "Long" and price <= pos["sl"]) or (d == "Short" and price >= pos["sl"]):
                daily_losses += 1
                del open_positions[symbol]
        time.sleep(MONITOR_INT)

def report_daily():
    total = daily_wins + daily_losses
    wr = round(daily_wins / total * 100, 1) if total > 0 else 0
    send_telegram(
        f"📊 *Daily Report*\n"
        f"Signals: {daily_signals}\n"
        f"✅ Wins: {daily_wins}\n"
        f"❌ Losses: {daily_losses}\n"
        f"🏆 Winrate: {wr}%"
    )

def monitor():
    last_hb = 0
    symbols = ["BTCUSDT","ETHUSDT","DOGEUSDT","BNBUSDT","XRPUSDT",
               "RENDERUSDT","TRUMPUSDT","FARTCOINUSDT","XLMUSDT",
               "SHIBUSDT","ADAUSDT","NOTUSDT","PROMUSDT","PENDLEUSDT"]
    while True:
        now = datetime.utcnow()
        hr = (now.hour + 3) % 24
        mn = now.minute

        if SLEEP_HOURS[0] <= hr < SLEEP_HOURS[1]:
            time.sleep(60)
            continue

        if time.time() - last_hb > HEARTBEAT_INT:
            send_telegram("🤖 Bot is running and checking signals...")
            last_hb = time.time()

        threads = []
        for sym in symbols:
            t = threading.Thread(target=check_and_alert, args=(sym,))
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
