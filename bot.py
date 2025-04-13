
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
open_positions = {}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        logging.error(f"Telegram exception: {e}")

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

def detect_strong_candle(row, threshold=0.7):
    body = abs(row['close'] - row['open'])
    candle_range = row['high'] - row['low']
    if candle_range == 0:
        return None
    ratio = body / candle_range
    if ratio > threshold:
        return 'bullish_marubozu' if row['close'] > row['open'] else 'bearish_marubozu'
    return None

def detect_engulfing(df):
    if len(df) < 2:
        return None
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    if prev['close'] < prev['open'] and curr['close'] > curr['open'] and curr['close'] > prev['open'] and curr['open'] < prev['close']:
        return 'bullish_engulfing'
    if prev['close'] > prev['open'] and curr['close'] < curr['open'] and curr['open'] > prev['close'] and curr['close'] < prev['open']:
        return 'bearish_engulfing'
    return None

def check_cooldown(symbol, direction):
    key = f"{symbol}_{direction}"
    last_time = last_signals.get(key)
    now = time.time()
    if last_time and (now - last_time < SIGNAL_COOLDOWN):
        return False
    last_signals[key] = now
    return True

def monitor_position(symbol, direction, entry, sl, tp1, tp2):
    df = get_data('15m', symbol)
    last_price = df['close'].iloc[-1]

    if direction == 'Long':
        if last_price >= tp2:
            send_telegram_message(f"🎯 {symbol} - TP2 Hit. Full target reached.")
            open_positions.pop(symbol, None)
        elif last_price >= tp1:
            send_telegram_message(f"✅ {symbol} - TP1 Hit. Partial target reached.")
        elif last_price <= sl:
            send_telegram_message(f"❌ {symbol} - SL Hit. Trade failed.")
            open_positions.pop(symbol, None)
    else:
        if last_price <= tp2:
            send_telegram_message(f"🎯 {symbol} - TP2 Hit. Full target reached.")
            open_positions.pop(symbol, None)
        elif last_price <= tp1:
            send_telegram_message(f"✅ {symbol} - TP1 Hit. Partial target reached.")
        elif last_price >= sl:
            send_telegram_message(f"❌ {symbol} - SL Hit. Trade failed.")
            open_positions.pop(symbol, None)

def analyze_symbol(symbol):
    df = get_data('15m', symbol)
    if len(df) < 30:
        return None

    df['EMA20'] = ta.ema(df['close'], length=20)
    df['EMA50'] = ta.ema(df['close'], length=50)
    df['rsi'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACDs'] = macd['MACDs_12_26_9']
    adx = ta.adx(df['high'], df['low'], df['close'])
    df['ADX'] = adx['ADX_14']
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'])

    candle = df.iloc[-1]
    signal_type = detect_strong_candle(candle) or detect_engulfing(df)
    if not signal_type:
        return None

    rsi_val = df['rsi'].iloc[-1]
    adx_val = df['ADX'].iloc[-1]
    entry = df['close'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    atr = max(atr, entry * MIN_PERCENT_RISK, MIN_ATR)

    ema_buy = candle['close'] > candle['EMA20'] and candle['EMA20'] > candle['EMA50']
    ema_sell = candle['close'] < candle['EMA20'] and candle['EMA20'] < candle['EMA50']

    confirmations = []
    if 'bullish' in signal_type and rsi_val >= 50:
        confirmations.append("RSI")
    if 'bearish' in signal_type and rsi_val <= 50:
        confirmations.append("RSI")
    if (df['MACD'].iloc[-1] > df['MACDs'].iloc[-1]) if 'bullish' in signal_type else (df['MACD'].iloc[-1] < df['MACDs'].iloc[-1]):
        confirmations.append("MACD")
    if adx_val > ADX_THRESHOLD:
        confirmations.append("ADX")
    if ('bullish' in signal_type and ema_buy) or ('bearish' in signal_type and ema_sell):
        confirmations.append("EMA")

    confidence = len(confirmations)
    direction = 'Long' if 'bullish' in signal_type and confidence >= 3 else 'Short' if 'bearish' in signal_type and confidence >= 3 else None

    if direction and not check_cooldown(symbol, direction):
        return None

    if direction:
        sl = entry - atr * ATR_MULTIPLIER_SL if direction == 'Long' else entry + atr * ATR_MULTIPLIER_SL
        tp1 = entry + atr * TP1_MULTIPLIER if direction == 'Long' else entry - atr * TP1_MULTIPLIER
        tp2 = entry + atr * TP2_MULTIPLIER if direction == 'Long' else entry - atr * TP2_MULTIPLIER
        rr = abs(tp1 - entry) / abs(entry - sl)
        signal = f"""
🚨 AI Signal Alert
Symbol: {symbol}
Signal: {'🟢 BUY' if direction == 'Long' else '🔴 SELL'}
Pattern: {signal_type}
Confirmed by: {', '.join(confirmations)}
Entry: {entry:.6f}
Stop Loss: {sl:.6f}
Target 1: {tp1:.6f}
Target 2: {tp2:.6f}
Leverage Est.: {rr:.2f}X
Signal Strength: {'🔥' * confidence}
"""
        open_positions[symbol] = (direction, entry, sl, tp1, tp2)
        return signal
    return None

def monitor():
    symbols = [
        "BTCUSDT", "ETHUSDT", "DOGEUSDT", "BNBUSDT", "XRPUSDT",
        "RENDERUSDT", "TRUMPUSDT", "FARTCOINUSDT", "XLMUSDT",
        "SHIBUSDT", "ADAUSDT", "NOTUSDT", "PROMUSDT"
    ]
    while True:
        now = datetime.utcnow()
        tehran_hour = (now.hour + 3) % 24
        if SLEEP_HOURS[0] <= tehran_hour < SLEEP_HOURS[1]:
            time.sleep(60)
            continue
        for sym in symbols:
            try:
                if sym in open_positions:
                    monitor_position(sym, *open_positions[sym])
                else:
                    msg = analyze_symbol(sym)
                    if msg:
                        send_telegram_message(msg)
            except Exception as e:
                logging.error(f"Error analyzing {sym}: {e}")
        time.sleep(CHECK_INTERVAL)

@app.route('/')
def home():
    return "✅ Bot is running."

if __name__ == '__main__':
    threading.Thread(target=monitor, daemon=True).start()
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))
