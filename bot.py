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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

last_alert_time = 0
last_heartbeat_time = 0

# -------------------------------
# ارسال پیام به تلگرام
# -------------------------------
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            logging.error(f"خطا در ارسال پیام: {response.text}")
    except Exception as e:
        logging.error(f"Exception در ارسال پیام تلگرام: {e}")

# -------------------------------
# تحلیل تکنیکال و الگوریتم‌ها
# -------------------------------
def is_ranging_market(df):
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
    return adx.iloc[-1] < 20

def detect_marubozu(row, threshold=0.9):
    body = abs(row['close'] - row['open'])
    candle_range = row['high'] - row['low']
    if candle_range == 0:
        return None
    body_ratio = body / candle_range
    if body_ratio > threshold:
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

def analyze_symbol(symbol, timeframe='15m'):
    df = get_data(timeframe, symbol)
    if len(df) < 3:
        return f"تحلیل {symbol}: داده کافی نیست."

    if is_ranging_market(df):
        return f"تحلیل {symbol}: بازار در حالت رنج است. سیگنال صادر نشد."

    rsi = ta.rsi(df['close'], length=14)
    df['rsi'] = rsi
    macd = ta.macd(df['close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACDs'] = macd['MACDs_12_26_9']
    adx = ta.adx(df['high'], df['low'], df['close'])
    df['ADX'] = adx['ADX_14']
    df['DI+'] = adx['DMP_14']
    df['DI-'] = adx['DMN_14']

    marubozu = detect_marubozu(df.iloc[-1])
    engulfing = detect_engulfing(df)
    rsi_val = df['rsi'].iloc[-1]
    adx_val = df['ADX'].iloc[-1]
    entry_price = df['close'].iloc[-1]
    atr = ta.atr(df['high'], df['low'], df['close']).iloc[-1]
    effective_risk = max(atr, entry_price * MIN_PERCENT_RISK)

    final_signal = None
    sl = tp1 = tp2 = tp3 = None

    if marubozu == 'bullish_marubozu' or engulfing == 'bullish_engulfing':
        if rsi_val < 65 and df['MACD'].iloc[-1] > df['MACDs'].iloc[-1] and adx_val > ADX_THRESHOLD:
            final_signal = 'Long'
    elif marubozu == 'bearish_marubozu' or engulfing == 'bearish_engulfing':
        if rsi_val > 35 and df['MACD'].iloc[-1] < df['MACDs'].iloc[-1] and adx_val > ADX_THRESHOLD:
            final_signal = 'Short'

    if final_signal == 'Long':
        sl = entry_price - effective_risk * ATR_MULTIPLIER_SL
        tp1 = entry_price + effective_risk * TP1_MULTIPLIER
        tp2 = entry_price + effective_risk * TP2_MULTIPLIER
        tp3 = entry_price + effective_risk * TP3_MULTIPLIER
    elif final_signal == 'Short':
        sl = entry_price + effective_risk * ATR_MULTIPLIER_SL
        tp1 = entry_price - effective_risk * TP1_MULTIPLIER
        tp2 = entry_price - effective_risk * TP2_MULTIPLIER
        tp3 = entry_price - effective_risk * TP3_MULTIPLIER

    if final_signal:
        return f"""
تحلیل بازار برای {symbol}:
- قیمت فعلی: {entry_price:.4f}
- RSI: {rsi_val:.2f}
- MACD: {df['MACD'].iloc[-1]:.4f} | Signal: {df['MACDs'].iloc[-1]:.4f}
- ADX: {adx_val:.2f}
- الگو: {marubozu or engulfing or 'N/A'}
- سیگنال: ورود به پوزیشن {final_signal}
نقطه ورود: {entry_price:.4f}
SL: {sl:.4f}
TP1: {tp1:.4f}
TP2: {tp2:.4f}
TP3: {tp3:.4f}
"""
    return f"تحلیل بازار برای {symbol}: سیگنالی یافت نشد."

def analyze_symbol_mtf(symbol):
    a5 = analyze_symbol(symbol, '5m')
    a15 = analyze_symbol(symbol, '15m')
    if 'Long' in a5 and 'Long' in a15:
        return a15
    if 'Short' in a5 and 'Short' in a15:
        return a15
    return f"تحلیل {symbol}: تایید چند تایم‌فریمی نشد."

def monitor():
    symbols = [
        "BTCUSDT", "ETHUSDT", "SHIBUSDT", "NEARUSDT",
        "SOLUSDT", "DOGEUSDT", "BNBUSDT", "MOODENGUSDT",
        "ZECUSDT", "ONEUSDT", "RSRUSDT", "HOTUSDT",
        "XLMUSDT", "SONICUSDT", "CAKEUSDT"
    ]
    while True:
        for sym in symbols:
            try:
                msg = analyze_symbol_mtf(sym)
                if "ورود به پوزیشن" in msg:
                    send_telegram_message(msg)
                else:
                    logging.info(f"{sym} — بدون سیگنال: {msg.strip()}")
            except Exception as e:
                logging.error(f"خطا در تحلیل {sym}: {e}")
        time.sleep(600)

@app.route('/')
def home():
    return "I'm alive!"

if __name__ == '__main__':
    threading.Thread(target=monitor, daemon=True).start()
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
