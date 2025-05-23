import os
import time
import logging
import threading
import requests
import pandas as pd
from flask import Flask, request
from datetime import datetime, timedelta

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
MONITOR_INT     = 120
SLEEP_HOURS     = (0, 7)  # UTC+3 hours sleep window

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
    logging.info(f"📨 Sending message to Telegram:\n{msg}")
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
    logging.info(f"✅ Fetched {len(df)} bars for {sym} ({tf}) from {df.index.min()} to {df.index.max()}")
    return df[["open","high","low","close","vol"]]

# ───── Indicators ─────
def pivot_high(df, lb):
    return df["high"].rolling(lb*2+1, center=True) \
             .apply(lambda x: x.iloc[lb]==x.max()).fillna(False)
def pivot_low(df, lb):
    return df["low"].rolling(lb*2+1, center=True) \
             .apply(lambda x: x.iloc[lb]==x.min()).fillna(False)
def rsi(series, length):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()
    rs = avg_gain/avg_loss
    return 100 - (100/(1+rs))

# ───── Cooldown ─────
def check_cooldown(sym, direction, idx):
    key = f"{sym}_{direction}"
    if last_signals.get(key)==idx:
        return False
    last_signals[key]=idx
    return True

# ───── Signal Analysis ─────
def analyze_symbol(sym, tf="15m"):
    global daily_signals
    logging.info(f"🔍 Analyzing {sym} on {tf}")
    df = get_data(tf, sym)
    if df is None or len(df)<PIVOT_LOOKBACK*2+1:
        logging.info(f"❌ Insufficient data for {sym}")
        return None
    df["EMA9"] = df["close"].ewm(span=EMA_LEN, adjust=False).mean()
    df["RSI"]  = rsi(df["close"], RSI_LEN)
    df["PH"]   = pivot_high(df, PIVOT_LOOKBACK)
    df["PL"]   = pivot_low(df, PIVOT_LOOKBACK)

    prev = df.iloc[-2]
    last = df.iloc[-1]
    idx  = last.name
    entry = last["close"]

    direction = None
    ob_type   = None
    early     = False

    # strict
    if prev["PL"] and entry>last["EMA9"] and last["RSI"]>RSI_BUY_LVL:
        direction, ob_type = "Long","Bull OB"
    elif prev["PH"] and entry<last["EMA9"] and last["RSI"]<RSI_SELL_LVL:
        direction, ob_type = "Short","Bear OB"
    # early
    elif prev["PL"] and entry>last["EMA9"]:
        early, ob_type = True, "Bull OB"
    elif prev["PH"] and entry<last["EMA9"]:
        early, ob_type = True, "Bear OB"
    else:
        logging.info(f"— No valid OB/EMA signal for {sym}")
        return None

    if not check_cooldown(sym, direction or "early", idx):
        logging.info(f"⏱️ Cooldown active for {sym}")
        return None

    if early:
        msg = (
            f"🟡 *Early Signal Alert*\n"
            f"*Symbol:* `{sym}`\n"
            f"*Potential:* {'🟢 BUY' if ob_type=='Bull OB' else '🔴 SELL'}\n"
            f"*Price:* `{entry:.6f}` | *RSI:* `{last['RSI']:.2f}`\n"
            f"*OB Type:* {ob_type}\n"
            f"🔍 Waiting RSI confirmation..."
        )
        logging.info(f"ℹ️ Early signal prepared for {sym}")
        return msg

    tr = pd.concat([
        df["high"]-df["low"],
        (df["high"]-df["close"].shift()).abs(),
        (df["low"]-df["close"].shift()).abs()],axis=1).max(axis=1)
    atr_val = tr.rolling(ATR_LEN).mean().iloc[-1]

    if direction=="Long":
        sl = entry-ATR_SL_MULT*atr_val
        tp1= entry+ATR_TP1_MULT*atr_val
        tp2= entry+ATR_TP2_MULT*atr_val
    else:
        sl = entry+ATR_SL_MULT*atr_val
        tp1= entry-ATR_TP1_MULT*atr_val
        tp2= entry-ATR_TP2_MULT*atr_val

    open_positions[sym] = {"dir":direction,"sl":sl,"tp1":tp1,"tp2":tp2}
    daily_signals += 1
    msg = (
        f"🚨 *AI Signal Alert*\n"
        f"*Symbol:* `{sym}`\n"
        f"*Signal:* {'🟢 BUY' if direction=='Long' else '🔴 SELL'}\n"
        f"*Type:* {ob_type}\n"
        f"*Price:* `{entry:.6f}`\n"
        f"*SL:* `{sl:.6f}`  *TP1:* `{tp1:.6f}`  *TP2:* `{tp2:.6f}`"
    )
    logging.info(f"✅ Final signal for {sym}: {direction} at {entry:.6f}")
    return msg

# ───── Alert Routine ─────
def check_and_alert(sym):
    logging.info(f"▶️ Checking {sym} (15m only)...")
    msg = analyze_symbol(sym, "15m")
    if msg:
        send_telegram(msg)
    else:
        logging.info(f"❌ No signal for {sym}")
    return msg

# ───── Position Monitoring ─────
def monitor_positions():
    global daily_wins,daily_losses
    while True:
        for sym,pos in list(open_positions.items()):
            df = get_data("15m",sym)
            if df is None: continue
            price=df["close"].iloc[-1]
            if pos["dir"]=="Long":
                if price>=pos["tp2"]: daily_wins+=1; del open_positions[sym]
                elif price<=pos["sl"]: daily_losses+=1; del open_positions[sym]
            else:
                if price<=pos["tp2"]: daily_wins+=1; del open_positions[sym]
                elif price>=pos["sl"]: daily_losses+=1; del open_positions[sym]
        time.sleep(MONITOR_INT)

# ───── Daily Report ─────
def report_daily():
    total=daily_wins+daily_losses
    wr=round(daily_wins/total*100,1) if total>0 else 0
    logging.info("🗒️ Sending daily report")
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

@app.route("/check",methods=["GET"])
def manual_check():
    sym=request.args.get("symbol","ETHUSDT").upper()
    res=check_and_alert(sym)
    return (res if res else f"No signal for {sym}"),200

# ───── Main Monitor Loop ─────
def monitor():
    last_hb=0
    symbols=[
        "BTCUSDT","ETHUSDT","DOGEUSDT","BNBUSDT","XRPUSDT",
        "RENDERUSDT","TRUMPUSPTUSDT","FARTCOINUSDT","XLMUSDT",
        "SHIBUSDT","ADAUSDT","NOTUSDT","PROMUSMT","PENDLEUSDT"
    ]
    while True:
        now=datetime.utcnow()
        mins=now.minute%15; secs=now.second
        wait_sec=(15-mins)*60-secs+2
        logging.info(f"⏳ Waiting {wait_sec}s until next 15m candle close")
        time.sleep(wait_sec)

        hr=(datetime.utcnow().hour+3)%24; mn=datetime.utcnow().minute
        if SLEEP_HOURS[0]<=hr<SLEEP_HOURS[1]:
            logging.info(f"😴 Within sleep hours ({hr}), skipping cycle")
            continue

        if time.time()-last_hb>HEARTBEAT_INT:
            logging.info("💓 Heartbeat: bot is alive")
            send_telegram("🤖 Bot live and scanning.")
            last_hb=time.time()

        threads=[]
        logging.info("🚀 Starting symbol checks...")
        for s in symbols:
            t=threading.Thread(target=check_and_alert,args=(s,))
            t.start(); threads.append(t)
        for t in threads: t.join()
        logging.info("✅ Cycle complete")

        if hr==23 and mn>=55:
            report_daily()

# ───── Main Operation ─────
if __name__=="__main__":
    # Notify on deployment
    send_telegram("🚀 Bot deployed and starting monitoring.")
    # Start background threads
    threading.Thread(target=monitor_positions, daemon=True).start()
    threading.Thread(target=monitor, daemon=True).start()
    port = int(os.getenv("PORT", 8080))
    logging.info(f"🔌 Starting Flask on port {port}")
    app.run(host="0.0.0.0", port=port)
