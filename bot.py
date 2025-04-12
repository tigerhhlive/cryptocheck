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

# تنظیمات
ATR_MULTIPLIER_SL = 1.2
TP1_MULTIPLIER = 1.0
TP2_MULTIPLIER = 2.0
MIN_PERCENT_RISK = 0.02
HEARTBEAT_INTERVAL = 7200
CHECK_INTERVAL = 600
SLEEP_HOURS = (0, 7)
RSI_LIMIT = 85

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            logging.error(f"خطا در ارسال پیام: {response.text}")
    except Exception as e:
        logging.error(f"Exception در ارسال پیام تلگرام: {e}")

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

def detect_candle_patterns(df):
    row = df.iloc[-1]
    prev = df.iloc[-2]
    body = abs(row['close'] - row['open'])
    candle_range = row['high'] - row['low']
    ratio = body / candle_range if candle_range != 0 else 0
    avg_range = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
    strong_bull = ratio > 0.65 and row['close'] > row['open'] and candle_range > avg_range
    strong_bear = ratio > 0.65 and row['close'] < row['open'] and candle_range > avg_range
    bullish_eng = prev['close'] < prev['open'] and row['close'] > row['open'] and row['close'] > prev['open'] and row['open'] < prev['close']
    bearish_eng = prev['close'] > prev['open'] and row['close'] < row['open'] and row['open'] > prev['close'] and row['close'] < prev['open']
    return strong_bull or bullish_eng, strong_bear or bearish_eng

def analyze_symbol(symbol, timeframe='15m'):
    df = get_data(timeframe, symbol)
    if len(df) < 20:
        return None

    df['EMA20'] = ta.ema(df['close'], length=20)
    df['EMA50'] = ta.ema(df['close'], length=50)
    df['EMA200'] = ta.ema(df['close'], length=200)
    df['rsi'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACDs'] = macd['MACDs_12_26_9']
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)

    row = df.iloc[-1]
    bull_cond, bear_cond = detect_candle_patterns(df)
    rsi = row['rsi']
    macd_ok_long = row['MACD'] > row['MACDs']
    macd_ok_short = row['MACD'] < row['MACDs']
    above_ema = row['close'] > row['EMA20'] and row['EMA20'] > row['EMA50']
    below_ema = row['close'] < row['EMA20'] and row['EMA20'] < row['EMA50']
    above_ema200 = row['close'] > row['EMA200']

    risk = max(row['atr'], row['close'] * MIN_PERCENT_RISK)
    entry = row['close']
    sl_long = entry - risk * ATR_MULTIPLIER_SL
    tp1_long = entry + risk * TP1_MULTIPLIER
    tp2_long = entry + risk * TP2_MULTIPLIER
    sl_short = entry + risk * ATR_MULTIPLIER_SL
    tp1_short = entry - risk * TP1_MULTIPLIER
    tp2_short = entry - risk * TP2_MULTIPLIER

    direction = None
    if bull_cond and rsi < RSI_LIMIT and macd_ok_long and (above_ema or above_ema200):
        direction = 'Long'
    elif bear_cond and rsi > 30 and macd_ok_short and below_ema:
        direction = 'Short'

    if direction:
        sl = sl_long if direction == 'Long' else sl_short
        tp1 = tp1_long if direction == 'Long' else tp1_short
        tp2 = tp2_long if direction == 'Long' else tp2_short
        rr_ratio = abs(tp1 - entry) / abs(entry - sl)
        return f"""
🚨 This Is AI Signal Alert . Ignore it 🚨
Symbol: {symbol}
Signal: {'BUY MARKET' if direction == 'Long' else 'SELL MARKET'}
Price: {entry:.6f}
Stop Loss: {sl:.6f}  
Target Level 1: {tp1:.6f}
Target Level 2: {tp2:.6f}
leverage : {rr_ratio:.2f}X
"""
    return None

def analyze_symbol_mtf(symbol):
    a5 = analyze_symbol(symbol, '5m')
    a15 = analyze_symbol(symbol, '15m')
    if a5 and a15 and (('BUY' in a5 and 'BUY' in a15) or ('SELL' in a5 and 'SELL' in a15)):
        return a15
    return None

def monitor():
    symbols = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "BNBUSDT", "XRPUSDT"]
    last_heartbeat = 0
    while True:
        now = datetime.utcnow()
        tehran_hour = (now.hour + 3) % 24
        if SLEEP_HOURS[0] <= tehran_hour < SLEEP_HOURS[1]:
            logging.info("ربات در حالت خواب شبانه است")
            time.sleep(60)
            continue

        if time.time() - last_heartbeat > HEARTBEAT_INTERVAL:
            send_telegram_message("🤖 ربات فعال است و در حال بررسی سیگنال‌ها می‌باشد")
            last_heartbeat = time.time()

        for sym in symbols:
            try:
                msg = analyze_symbol_mtf(sym)
                if msg:
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
