# ✅ Crypto Signal Bot - Refined Version
# Updated based on full review and debugging
# Improvements: Better entry detection, SL/TP logic, and cleaner signal filtering
# Created by ChatGPT for professional trading

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
ATR_MULTIPLIER_SL = 1.5
TP1_MULTIPLIER = 2.0
TP2_MULTIPLIER = 3.0
MIN_PERCENT_RISK = 0.005
MAX_SL_PERCENT = 0.03
HEARTBEAT_INTERVAL = 7200
CHECK_INTERVAL = 600
SLEEP_HOURS = (0, 7)
SIGNAL_COOLDOWN = 1800

last_signals = {}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        logging.error(f"Telegram Error: {e}")

def get_data(timeframe, symbol):
    aggregate = 5 if timeframe == '5m' else 15
    url = "https://min-api.cryptocompare.com/data/v2/histominute"
    params = {
        'fsym': symbol[:-4],
        'tsym': "USDT",
        'limit': 60,
        'aggregate': aggregate,
        'api_key': CRYPTOCOMPARE_API_KEY
    }
    res = requests.get(url, params=params, timeout=10)
    data = res.json()['Data']['Data']
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
    df['volume'] = df['volumeto']
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

def detect_marubozu(candle, threshold=0.75):
    body = abs(candle['close'] - candle['open'])
    rng = candle['high'] - candle['low']
    if rng == 0: return None
    ratio = body / rng
    if ratio > threshold:
        return 'bullish_marubozu' if candle['close'] > candle['open'] else 'bearish_marubozu'
    return None

def check_conditions(df, signal_type):
    candle = df.iloc[-1]
    rsi = ta.rsi(df['close']).iloc[-1]
    macd = ta.macd(df['close'])
    adx = ta.adx(df['high'], df['low'], df['close'])

    ema20 = ta.ema(df['close'], length=20)
    ema50 = ta.ema(df['close'], length=50)

    df['EMA20'] = ema20
    df['EMA50'] = ema50

    confirmations = []
    if ('bullish' in signal_type and rsi >= 50) or ('bearish' in signal_type and rsi <= 50):
        confirmations.append('RSI')
    if (macd['MACD_12_26_9'].iloc[-1] > macd['MACDs_12_26_9'].iloc[-1]) if 'bullish' in signal_type else        (macd['MACD_12_26_9'].iloc[-1] < macd['MACDs_12_26_9'].iloc[-1]):
        confirmations.append('MACD')
    if adx['ADX_14'].iloc[-1] > ADX_THRESHOLD:
        confirmations.append('ADX')
    if ('bullish' in signal_type and candle['close'] > ema20.iloc[-1] > ema50.iloc[-1]) or        ('bearish' in signal_type and candle['close'] < ema20.iloc[-1] < ema50.iloc[-1]):
        confirmations.append('EMA')
    return confirmations

def check_cooldown(symbol, direction):
    key = f"{symbol}_{direction}"
    now = time.time()
    if key in last_signals and now - last_signals[key] < SIGNAL_COOLDOWN:
        return False
    last_signals[key] = now
    return True

def analyze_symbol(symbol, timeframe='15m'):
    df = get_data(timeframe, symbol)
    if len(df) < 30:
        return None

    candle = df.iloc[-2]  # take the previous candle, not the current forming one
    entry_price = candle['close']
    atr = ta.atr(df['high'], df['low'], df['close']).iloc[-2]
    atr = max(atr, entry_price * MIN_PERCENT_RISK)
    sl_raw = atr * ATR_MULTIPLIER_SL
    sl = entry_price - sl_raw if candle['close'] > candle['open'] else entry_price + sl_raw

    if abs(entry_price - sl) / entry_price > MAX_SL_PERCENT:
        return None  # skip weird SL values

    signal_type = detect_marubozu(candle)
    if not signal_type:
        return None

    confirmations = check_conditions(df, signal_type)
    direction = 'Long' if 'bullish' in signal_type and len(confirmations) >= 3 else                 'Short' if 'bearish' in signal_type and len(confirmations) >= 3 else None

    if not direction or not check_cooldown(symbol, direction):
        return None

    tp1 = entry_price + atr * TP1_MULTIPLIER if direction == 'Long' else entry_price - atr * TP1_MULTIPLIER
    tp2 = entry_price + atr * TP2_MULTIPLIER if direction == 'Long' else entry_price - atr * TP2_MULTIPLIER
    rr_ratio = round(abs(tp1 - entry_price) / abs(entry_price - sl), 2)
    confidence = "🔥" * len(confirmations)

    msg = f"""🚨 *AI Signal Alert*
*Symbol:* `{symbol}`
*Signal:* {'🟢 BUY MARKET' if direction == 'Long' else '🔴 SELL MARKET'}
*Entry:* `{entry_price:.6f}`
*Stop Loss:* `{sl:.6f}`
*Target 1:* `{tp1:.6f}`
*Target 2:* `{tp2:.6f}`
*Leverage (est.):* `{rr_ratio:.2f}X`
*Confirmed by:* {", ".join(confirmations)}
*Signal Strength:* {confidence}
"""
    return msg

def monitor():
    symbols = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "BNBUSDT", "XRPUSDT",
        "RENDERUSDT", "TRUMPUSDT", "FARTCOINUSDT", "XLMUSDT",
        "SHIBUSDT", "ADAUSDT", "NOTUSDT", "PROMUSDT"]

    while True:
        for sym in symbols:
            try:
                msg = analyze_symbol(sym)
                if msg:
                    logging.info(f"✅ Signal sent: {sym}")
                    send_telegram_message(msg)
            except Exception as e:
                logging.error(f"Error analyzing {sym}: {e}")
        time.sleep(CHECK_INTERVAL)

@app.route('/')
def home():
    return "✅ Crypto Signal Bot is running."

if __name__ == '__main__':
    threading.Thread(target=monitor, daemon=True).start()
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
