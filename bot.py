
# Crypto Signal Bot with Breakout Spike Detection (Enhanced Version)
import os
import time
import logging
import requests
import threading
import pandas as pd
import pandas_ta as ta
from flask import Flask
from datetime import datetime

app = Flask(__name__)

CRYPTOCOMPARE_API_KEY = os.environ.get('CRYPTOCOMPARE_API_KEY')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

ADX_THRESHOLD = 20
ATR_PERIOD = 14
ATR_MULTIPLIER_SL = 1.2
TP1_MULTIPLIER = 1.8
TP2_MULTIPLIER = 2.8
MIN_PERCENT_RISK = 0.03
HEARTBEAT_INTERVAL = 7200
CHECK_INTERVAL = 600
SLEEP_HOURS = (0, 7)
MIN_ATR = 0.001
SIGNAL_COOLDOWN = 1800

last_signals = {}
daily_signal_count = 0
daily_hit_count = 0
last_report_day = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        logging.error(f"Telegram error: {e}")

def get_data(timeframe, symbol):
    url = "https://min-api.cryptocompare.com/data/v2/histominute"
    aggregate = 5 if timeframe == '5m' else 15
    limit = 60
    fsym, tsym = symbol[:-4], "USDT"
    params = {
        'fsym': fsym,
        'tsym': tsym,
        'limit': limit,
        'aggregate': aggregate,
        'api_key': CRYPTOCOMPARE_API_KEY
    }
    res = requests.get(url, params=params, timeout=10)
    data = res.json()['Data']['Data']
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
    df['volume'] = df['volumeto']
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

def check_cooldown(symbol, direction):
    key = f"{symbol}_{direction}"
    now = time.time()
    if key in last_signals and now - last_signals[key] < SIGNAL_COOLDOWN:
        return False
    last_signals[key] = now
    return True

def analyze_spike_algo(df, symbol):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    body = abs(last['close'] - last['open'])
    prev_body = abs(prev['close'] - prev['open'])
    spike = body > prev_body * 2.5
    rsi = ta.rsi(df['close'], length=14).iloc[-1]
    atr = ta.atr(df['high'], df['low'], df['close'], length=14).iloc[-1]
    if spike and rsi > 60 and check_cooldown(symbol, 'Long'):
        entry = last['close']
        sl = entry - atr * 1.2
        tp1 = entry + atr * 1.8
        tp2 = entry + atr * 2.8
        rr = abs(tp1 - entry) / abs(entry - sl)
        return f"""⚡️ *Breakout Spike Detected!*
*Symbol:* `{symbol}`
*Signal:* 🟢 *BUY MARKET*
*Entry:* `{entry:.6f}`
*Stop Loss:* `{sl:.6f}`
*Target 1:* `{tp1:.6f}`
*Target 2:* `{tp2:.6f}`
*Leverage Est.:* `{rr:.2f}X`
*Reason:* Strong spike with RSI>60
""", True
    return None, False

def analyze_symbol_mtf(symbol):
    df_15m = get_data('15m', symbol)
    spike_msg, valid = analyze_spike_algo(df_15m, symbol)
    if valid:
        return spike_msg, None
    return None, "No spike breakout"

def monitor():
    symbols = [
        "BTCUSDT", "ETHUSDT", "DOGEUSDT", "BNBUSDT", "XRPUSDT",
        "RENDERUSDT", "TRUMPUSDT", "FARTCOINUSDT", "XLMUSDT",
        "SHIBUSDT", "ADAUSDT", "NOTUSDT", "PROMUSDT"
    ]
    last_heartbeat = 0
    while True:
        now = datetime.utcnow()
        tehran_hour = (now.hour + 3) % 24
        if SLEEP_HOURS[0] <= tehran_hour < SLEEP_HOURS[1]:
            logging.info("ربات در حالت خواب شبانه است")
            time.sleep(60)
            continue
        if time.time() - last_heartbeat > HEARTBEAT_INTERVAL:
            send_telegram_message("🤖 ربات فعال است.")
            last_heartbeat = time.time()
        for sym in symbols:
            try:
                msg, reason = analyze_symbol_mtf(sym)
                if msg:
                    logging.info(f"SIGNAL DETECTED FOR {sym}")
                    send_telegram_message(msg)
            except Exception as e:
                logging.error(f"Error analyzing {sym}: {e}")
        time.sleep(CHECK_INTERVAL)

@app.route('/')
def home():
    return "I'm alive!"

if __name__ == '__main__':
    threading.Thread(target=monitor, daemon=True).start()
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
