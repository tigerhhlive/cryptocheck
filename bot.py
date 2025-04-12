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

# تنظیمات کلی
ATR_PERIOD = 14
TP1_MULTIPLIER = 1.0
TP2_MULTIPLIER = 2.0
MIN_PERCENT_RISK = 0.03
HEARTBEAT_INTERVAL = 7200
CHECK_INTERVAL = 600
SLEEP_HOURS = (0, 7)  # ساعت ایران

symbols = [
    "BTCUSDT", "ETHUSDT", "DOGEUSDT", "BNBUSDT", "XRPUSDT",
    "RENDERUSDT", "TRUMPUSDT", "FARTCOINUSDT", "XLMUSDT",
    "SHIBUSDT", "ADAUSDT", "NOTUSDT"
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        logging.error(f"تلگرام خطا: {e}")


def get_data(symbol, timeframe='15m'):
    url = "https://min-api.cryptocompare.com/data/v2/histominute"
    aggregate = 5 if timeframe == '5m' else 15
    fsym = symbol.replace("USDT", "")
    params = {
        'fsym': fsym,
        'tsym': 'USDT',
        'limit': 60,
        'aggregate': aggregate,
        'api_key': CRYPTOCOMPARE_API_KEY
    }
    res = requests.get(url, params=params)
    data = res.json()['Data']['Data']
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
    df['volume'] = df['volumeto']
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]


def analyze_symbol(symbol):
    df = get_data(symbol)
    if len(df) < 30:
        return None

    df['EMA20'] = ta.ema(df['close'], 20)
    df['EMA50'] = ta.ema(df['close'], 50)
    df['EMA200'] = ta.ema(df['close'], 200)
    df['RSI'] = ta.rsi(df['close'], 14)
    macd = ta.macd(df['close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACDs'] = macd['MACDs_12_26_9']
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'])
    df['Range'] = df['high'] - df['low']
    df['Body'] = abs(df['close'] - df['open'])
    df['Ratio'] = df['Body'] / df['Range']

    candle = df.iloc[-1]
    cond_bull = candle['Ratio'] > 0.65 and candle['close'] > candle['open'] and candle['Range'] > df['Range'].mean()
    cond_bear = candle['Ratio'] > 0.65 and candle['close'] < candle['open'] and candle['Range'] > df['Range'].mean()

    macd_cross_up = candle['MACD'] > candle['MACDs']
    macd_cross_down = candle['MACD'] < candle['MACDs']

    risk = max(candle['ATR'], candle['close'] * MIN_PERCENT_RISK)
    direction = None
    reason = []

    if cond_bull and macd_cross_up and candle['RSI'] < 85 and candle['close'] > candle['EMA50']:
        direction = 'Long'
    elif cond_bear and macd_cross_down and candle['RSI'] > 30 and candle['close'] < candle['EMA50']:
        direction = 'Short'
    else:
        if not cond_bull and not cond_bear:
            reason.append("❌ No strong candle")
        if not macd_cross_up and not macd_cross_down:
            reason.append("❌ MACD not crossed")
        if not (candle['close'] > candle['EMA50'] or candle['close'] < candle['EMA50']):
            reason.append("❌ EMA alignment")
        if not reason:
            reason.append("🔍 Not confirmed conditions")

    if direction:
        entry = candle['close']
        sl = entry - risk if direction == 'Long' else entry + risk
        tp1 = entry + TP1_MULTIPLIER * risk if direction == 'Long' else entry - TP1_MULTIPLIER * risk
        tp2 = entry + TP2_MULTIPLIER * risk if direction == 'Long' else entry - TP2_MULTIPLIER * risk
        rr = abs(tp1 - entry) / abs(entry - sl)
        return f"""
🚨 AI Adaptive Signal 🚨
Symbol: {symbol}
Signal: {'BUY MARKET' if direction == 'Long' else 'SELL MARKET'}
Price: {entry:.6f}
Stop Loss: {sl:.6f}
Target 1: {tp1:.6f}
Target 2: {tp2:.6f}
Leverage: {rr:.2f}X
"""
    else:
        return f"⚠️ {symbol} - No Signal\n" + "\n".join(reason)


def monitor():
    last_heartbeat = 0
    while True:
        now = datetime.utcnow()
        tehran_hour = (now.hour + 3) % 24
        if SLEEP_HOURS[0] <= tehran_hour < SLEEP_HOURS[1]:
            time.sleep(60)
            continue

        if time.time() - last_heartbeat > HEARTBEAT_INTERVAL:
            send_telegram_message("✅ ربات فعال است.")
            last_heartbeat = time.time()

        for sym in symbols:
            try:
                msg = analyze_symbol(sym)
                if msg:
                    send_telegram_message(msg)
            except Exception as e:
                logging.error(f"Error analyzing {sym}: {e}")
        time.sleep(CHECK_INTERVAL)

@app.route('/')
def home():
    return "✅ AI Signal Bot Running"

if __name__ == '__main__':
    threading.Thread(target=monitor, daemon=True).start()
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
