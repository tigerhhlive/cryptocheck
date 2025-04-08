import os
import time
import logging
import requests
from statistics import mean, stdev
import threading
import pandas as pd
import pandas_ta as ta
from flask import Flask

app = Flask(__name__)

CRYPTOCOMPARE_API_KEY = os.environ.get('CRYPTOCOMPARE_API_KEY')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

NUM_CANDLES = 60
VOLUME_MULTIPLIER = 1.2
PRICE_CHANGE_THRESHOLD = 0.8
STD_MULTIPLIER = 1.0
ALERT_COOLDOWN = 900
HEARTBEAT_INTERVAL = 3600

ADX_THRESHOLD = 25
ATR_PERIOD = 14
ATR_MULTIPLIER_SL = 1.5
TP1_MULTIPLIER = 1.0
TP2_MULTIPLIER = 1.5
TP3_MULTIPLIER = 2.0
MIN_PERCENT_RISK = 0.05

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ارسال پیام به تلگرام فقط در صورت وجود سیگنال
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            logging.error(f"خطا در ارسال پیام: {response.text}")
    except Exception as e:
        logging.error(f"Exception در ارسال پیام تلگرام: {e}")

# دریافت داده و اندیکاتورها
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

def is_ranging_market(df):
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
    return adx.iloc[-1] < 20

def detect_strong_candle(row, threshold=0.9):
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

def analyze_symbol(symbol, timeframe='15m'):
    df = get_data(timeframe, symbol)
    if len(df) < 3:
        return None

    if is_ranging_market(df):
        return None

    df['EMA20'] = ta.ema(df['close'], length=20)
    df['EMA50'] = ta.ema(df['close'], length=50)

    rsi = ta.rsi(df['close'], length=14)
    df['rsi'] = rsi
    macd = ta.macd(df['close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACDs'] = macd['MACDs_12_26_9']
    adx = ta.adx(df['high'], df['low'], df['close'])
    df['ADX'] = adx['ADX_14']
    df['DI+'] = adx['DMP_14']
    df['DI-'] = adx['DMN_14']

    candle = df.iloc[-1]
    signal_type = detect_strong_candle(candle) or detect_engulfing(df)
    rsi_val = df['rsi'].iloc[-1]
    adx_val = df['ADX'].iloc[-1]
    entry = df['close'].iloc[-1]
    atr = ta.atr(df['high'], df['low'], df['close']).iloc[-1]
    risk = max(atr, entry * MIN_PERCENT_RISK)

    above_ema = candle['close'] > candle['EMA20'] and candle['EMA20'] > candle['EMA50']
    below_ema = candle['close'] < candle['EMA20'] and candle['EMA20'] < candle['EMA50']

    direction = None
    if signal_type == 'bullish_marubozu' or signal_type == 'bullish_engulfing':
        if rsi_val < 65 and df['MACD'].iloc[-1] > df['MACDs'].iloc[-1] and adx_val > ADX_THRESHOLD and above_ema:
            direction = 'Long'
    elif signal_type == 'bearish_marubozu' or signal_type == 'bearish_engulfing':
        if rsi_val > 35 and df['MACD'].iloc[-1] < df['MACDs'].iloc[-1] and adx_val > ADX_THRESHOLD and below_ema:
            direction = 'Short'

    if direction:
        if direction == 'Long':
            sl = entry - risk * ATR_MULTIPLIER_SL
            tp1 = entry + risk * TP1_MULTIPLIER
            tp2 = entry + risk * TP2_MULTIPLIER
        else:
            sl = entry + risk * ATR_MULTIPLIER_SL
            tp1 = entry - risk * TP1_MULTIPLIER
            tp2 = entry - risk * TP2_MULTIPLIER

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
    symbols = [
        "BTCUSDT", "ETHUSDT", "SHIBUSDT", "NEARUSDT", "SOLUSDT", "DOGEUSDT",
        "BNBUSDT", "MOODENGUSDT", "ZECUSDT", "ONEUSDT", "RSRUSDT",
        "HOTUSDT", "XLMUSDT", "SONICUSDT", "CAKEUSDT"
    ]
    while True:
        for sym in symbols:
            try:
                msg = analyze_symbol_mtf(sym)
                if msg:
                    send_telegram_message(msg)
            except Exception as e:
                logging.error(f"Error analyzing {sym}: {e}")
        time.sleep(600)

@app.route('/')
def home():
    return "I'm alive!"

if __name__ == '__main__':
    threading.Thread(target=monitor, daemon=True).start()
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
