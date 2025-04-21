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

# ───── ENV Settings ─────
CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")
TELEGRAM_BOT_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID       = os.getenv("TELEGRAM_CHAT_ID")

# ───── Strategy Parameters ─────
EMA_LEN        = 9
ATR_LEN        = 14
ATR_SL_MULT    = 1.2
ATR_TP1_MULT   = 1.5
ATR_TP2_MULT   = 2.5
RSI_LEN        = 14
RSI_BUY_LVL    = 30
RSI_SELL_LVL   = 70
PIVOT_LOOKBACK = 5
SIGNAL_COOLDOWN= 1800
HEARTBEAT_INT  = 7200
CHECK_INT      = 600
MONITOR_INT    = 120
SLEEP_HOURS    = (0, 7)

# ───── Tracking ─────
last_signals   = {}
open_positions = {}
daily_signals  = 0
daily_wins     = 0
daily_losses   = 0

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(message)s")

def send_telegram(msg: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    r = requests.post(url, json=payload)
    if r.status_code != 200:
        logging.error(f"Telegram error: {r.text}")

def get_data(tf: str, sym: str) -> pd.DataFrame:
    agg = 5 if tf=="5m" else 15
    res = requests.get(
        "https://min-api.cryptocompare.com/data/v2/histominute",
        params={"fsym": sym[:-4], "tsym":"USDT", "limit":100, "aggregate":agg, "api_key":CRYPTOCOMPARE_API_KEY},
        timeout=10
    ).json()
    df = pd.DataFrame(res["Data"]["Data"])
    df["timestamp"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("timestamp", inplace=True)
    df[["open","high","low","close","vol"]] = df[["open","high","low","close","volumeto"]]
    return df[["open","high","low","close","vol"]]

def pivot_high(df, lb):
    ph = df['high'].rolling(window=lb*2+1, center=True).apply(lambda x: 1 if x.iloc[lb]==x.max() else 0)
    return ph == 1

def pivot_low(df, lb):
    pl = df['low'].rolling(window=lb*2+1, center=True).apply(lambda x: 1 if x.iloc[lb]==x.min() else 0)
    return pl == 1

# 🔧 Cooldown غیرفعال برای تست
def check_cooldown(sym, direction, idx):
    return True

def analyze_symbol(sym, tf="15m"):
    global daily_signals
    df = get_data(tf, sym)
    if len(df) < PIVOT_LOOKBACK*2+5:
        return None

    df["EMA9"] = ta.ema(df["close"], length=EMA_LEN)
    df["ATR"]  = ta.atr(df["high"], df["low"], df["close"], length=ATR_LEN)
    df["RSI"]  = ta.rsi(df["close"], length=RSI_LEN)
    df["PH"]   = pivot_high(df, PIVOT_LOOKBACK)
    df["PL"]   = pivot_low(df, PIVOT_LOOKBACK)

    last = df.iloc[-1]
    idx  = df.index[-1]
    direction, entry, rsiVal, atr = None, last["close"], last["RSI"], last["ATR"]
    early = False

    if last["PL"] and last["close"] > last["EMA9"]:
        direction = "Long" if rsiVal > RSI_BUY_LVL else None
        early = True if direction is None else False
    elif last["PH"] and last["close"] < last["EMA9"]:
        direction = "Short" if rsiVal < RSI_SELL_LVL else None
        early = True if direction is None else False
    else:
        logging.info(f"{sym}: No OB/EMA/RSI signal")
        return None

    if not check_cooldown(sym, direction or "early", idx):
        logging.info(f"{sym}: Cooldown active")
        return None

    sl,tp1,tp2 = None,None,None
    if direction == "Long":
        sl  = entry - atr*ATR_SL_MULT
        tp1 = entry + atr*ATR_TP1_MULT
        tp2 = entry + atr*ATR_TP2_MULT
    elif direction == "Short":
        sl  = entry + atr*ATR_SL_MULT
        tp1 = entry - atr*ATR_TP1_MULT
        tp2 = entry - atr*ATR_TP2_MULT

    if early:
        msg = (
            f"🟡 *Early Signal Alert (Watch Only)*\n"
            f"*Symbol:* `{sym}`\n"
            f"*Potential Direction:* {'🟢 BUY' if 'Long' in str(direction) else '🔴 SELL'}\n"
            f"*Price:* `{entry:.6f}` | *RSI:* `{rsiVal:.2f}`\n"
            f"🔍 Waiting RSI confirmation..."
        )
        return msg

    daily_signals += 1
    stars = "🔥🔥🔥"
    msg = (
        f"🚨 *AI Signal Alert*\n"
        f"*Symbol:* `{sym}`\n"
        f"*Signal:* {'🟢 BUY' if direction=='Long' else '🔴 SELL'}\n"
        f"*Price:* `{entry:.6f}`\n"
        f"*SL:* `{sl:.6f}`  *TP1:* `{tp1:.6f}`  *TP2:* `{tp2:.6f}`\n"
        f"*EMA9:* `{last['EMA9']:.4f}`  |  *RSI:* `{rsiVal:.2f}`\n"
        f"*Strength:* {stars}"
    )

    open_positions[sym] = {"dir":direction, "sl":sl, "tp1":tp1, "tp2":tp2}
    logging.info(f"{sym}: Signal {direction} @ {entry:.6f}")
    return msg

def check_and_alert(sym):
    logging.info(f"🔍 Checking {sym}...")
    msg = analyze_symbol(sym, "15m")  # بدون MTF
    if msg:
        send_telegram(msg)

def monitor_positions():
    global daily_wins, daily_losses
    while True:
        for sym, pos in list(open_positions.items()):
            df = get_data("15m", sym)
            price = df["close"].iloc[-1]
            d = pos["dir"]
            if (d=="Long" and price>=pos["tp2"]) or (d=="Short" and price<=pos["tp2"]):
                daily_wins += 1
                del open_positions[sym]
            elif (d=="Long" and price<=pos["sl"]) or (d=="Short" and price>=pos["sl"]):
                daily_losses += 1
                del open_positions[sym]
        time.sleep(MONITOR_INT)

def report_daily():
    total = daily_wins + daily_losses
    wr    = round(daily_wins/total*100,1) if total>0 else 0
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
        hr = (now.hour+3)%24
        mn = now.minute

        if SLEEP_HOURS[0] <= hr < SLEEP_HOURS[1]:
            time.sleep(60); continue

        if time.time()-last_hb > HEARTBEAT_INT:
            send_telegram("🤖 Bot live and scanning.")
            last_hb = time.time()

        threads = []
        for s in symbols:
            t = threading.Thread(target=check_and_alert, args=(s,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        if hr==23 and mn>=55:
            report_daily()

        time.sleep(CHECK_INT)

@app.route("/")
def home():
    return "✅ Crypto Signal Bot is running."

if __name__=="__main__":
    threading.Thread(target=monitor, daemon=True).start()
    threading.Thread(target=monitor_positions, daemon=True).start()
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
