import os
import time
import logging
import threading
import requests
import pandas as pd
from flask import Flask, request
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

# ───── Logging ─────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(message)s")
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# ───── Telegram Sender ─────
def send_telegram(msg: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload)
        if r.status_code != 200:
            logging.error(f"Telegram error: {r.text}")
    except Exception as e:
        logging.error(f"Error sending telegram: {e}")

# ───── Data Fetching ─────
def get_data(tf: str, sym: str) -> pd.DataFrame:
    agg = 5 if tf == "5m" else 15
    try:
        res = requests.get(
            "https://min-api.cryptocompare.com/data/v2/histominute",
            params={"fsym": sym[:-4], "tsym": "USDT", "limit": 200, "aggregate": agg, "api_key": CRYPTOCOMPARE_API_KEY},
            timeout=10
        ).json()
    except Exception as e:
        logging.error(f"Request error for {sym}: {e}")
        return None
    if res.get("Response") != "Success":
        logging.error(f"API error for {sym}: {res.get('Message')}")
        return None
    data = res.get("Data", {}).get("Data", [])
    if not data:
        logging.error(f"No data points for {sym}")
        return None
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("timestamp", inplace=True)
    df.rename(columns={"volumeto": "vol"}, inplace=True)
    return df[["open","high","low","close","vol"]]

# ───── Indicators ─────
def pivot_high(df, lb):
    return df["high"].rolling(window=lb*2+1, center=True) \
             .apply(lambda x: x.iloc[lb] == x.max()).fillna(False)

def pivot_low(df, lb):
    return df["low"].rolling(window=lb*2+1, center=True) \
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
    if df is None or len(df) < PIVOT_LOOKBACK*2+1:
        return None
    # Compute indicators
    df["EMA9"] = df["close"].ewm(span=EMA_LEN, adjust=False).mean()
    df["RSI"]  = rsi(df["close"], RSI_LEN)
    df["PH"]   = pivot_high(df, PIVOT_LOOKBACK)
    df["PL"]   = pivot_low(df, PIVOT_LOOKBACK)

    # Use previous bar pivot flags
    prev = df.iloc[-2]
    last = df.iloc[-1]
    idx  = last.name
    entry = last["close"]
    direction = None
    ob_type = None
    early = False

    # Strict signal
    if prev["PL"] and entry > last["EMA9"] and last["RSI"] > RSI_BUY_LVL:
        direction, ob_type = "Long", "Bull OB"
    elif prev["PH"] and entry < last["EMA9"] and last["RSI"] < RSI_SELL_LVL:
        direction, ob_type = "Short", "Bear OB"
    # Early signal
    elif prev["PL"] and entry > last["EMA9"]:
        early, ob_type = True, "Bull OB"
    elif prev["PH"] and entry < last["EMA9"]:
        early, ob_type = True, "Bear OB"
    else:
        return None

    if not check_cooldown(sym, direction or "early", idx):
        return None

    if early:
        return (
            f"🟡 *Early Signal Alert*\n"
            f"*Symbol:* `{sym}`\n"
            f"*Potential:* {'🟢 BUY' if ob_type=='Bull OB' else '🔴 SELL'}\n"
            f"*Price:* `{entry:.6f}` | *RSI:* `{last['RSI']:.2f}`\n"
            f"*OB Type:* {ob_type}\n"
            f"🔍 Waiting RSI confirmation..."
        )

    # Approximate ATR via true range rolling mean
    tr = pd.concat([df['high'] - df['low'],
                    (df['high'] - df['close'].shift()).abs(),
                    (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
    atr_val = tr.rolling(ATR_LEN).mean().iloc[-1]

    if direction == "Long":
        sl = entry - ATR_SL_MULT * atr_val
        tp1 = entry + ATR_TP1_MULT * atr_val
        tp2 = entry + ATR_TP2_MULT * atr_val
    else:
        sl = entry + ATR_SL_MULT * atr_val
        tp1 = entry - ATR_TP1_MULT * atr_val
        tp2 = entry - ATR_TP2_MULT * atr_val

    open_positions[sym] = {"dir": direction, "sl": sl, "tp1": tp1, "tp2": tp2}
    daily_signals += 1
    return (
        f"🚨 *AI Signal Alert*\n"
        f"*Symbol:* `{sym}`\n"
        f"*Signal:* {'🟢 BUY' if direction=='Long' else '🔴 SELL'}\n"
        f"*Type:* {ob_type}\n"
        f"*Price:* `{entry:.6f}`\n"
        f"*SL:* `{sl:.6f}`  *TP1:* `{tp1:.6f}`  *TP2:* `{tp2:.6f}`"
    )

# ───── Alert Routine ─────
def check_and_alert(sym):
    logging.info(f"🔍 Checking {sym} (15m only)...")
    msg = analyze_symbol(sym, "15m")
    if msg:
        send_telegram(msg)
        return msg
    return None

# ───── Position Monitoring ─────
def monitor_positions():
    global daily_wins, daily_losses
    while True:
        for sym, pos in list(open_positions.items()):
            df = get_data("15m", sym)
            if df is None:
                continue
            price = df["close"].iloc[-1]
            if pos["dir"] == "Long":
                if price >= pos["tp2"]:
                    daily_wins += 1
                    del open_positions[sym]
                elif price <= pos["sl"]:
                    daily_losses += 1
                    del open_positions[sym]
            else:
                if price <= pos["tp2"]:
                    daily_wins += 1
                    del open_positions[sym]
                elif price >= pos["sl"]:
                    daily_losses += 1
                    del open_positions[sym]
        time.sleep(MONITOR_INT)

# ───── Daily Report ─────
def report_daily():
    total = daily_wins + daily_losses
    wr = round(daily_wins/total*100,1) if total>0 else 0
    send_telegram(
        f"📊 *Daily Report*\n"
        f"Signals: {daily_signals}\n"
        f"✅ Wins: {daily_wins}\n"
        f"❌ Losses: {daily_losses}\n"
        f"🏆 Winrate: {wr}%"
    )

# ───── HTTP Endpoints ─────
@app.route("/")
def home():
    return "✅ Crypto Signal Bot is running."

@app.route("/check", methods=["GET"] )
def manual_check():
    symbol = request.args.get("symbol", "ETHUSDT").upper()
    result = check_and_alert(symbol)
    return (result if result else f"No signal for {symbol}"), 200

# ───── Main Monitor Loop ─────
def monitor():
    last_hb = 0
    symbols = [
        "BTCUSDT","ETHUSDT","DOGEUSDT","BNBUSDT","XRPUSDT",
        "RENDERUSDT","TRUMPUSDT","FARTCOINUSDT","XLMUSDT",
        "SHIBUSDT","ADAUSDT","NOTUSDT","PROMUSDT","PENDLEUSDT",
        "CETUSUSDT","MAGICUSDT","SOLVUSDT","ENAUSDT",
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

# ───── Main Operation ─────
if __name__=="__main__":
    threading.Thread(target=monitor_positions, daemon=True).start()
    threading.Thread(target=monitor, daemon=True).start()
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
