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

ADX_THRESHOLD = 25
ATR_PERIOD = 14
ATR_MULTIPLIER_SL = 1.2
TP1_MULTIPLIER = 0.8
TP2_MULTIPLIER = 1.2
MIN_PERCENT_RISK = 0.03
HEARTBEAT_INTERVAL = 7200
CHECK_INTERVAL = 600
SLEEP_HOURS = (0, 7)
MIN_ATR = 0.001

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log_file = open("ai_signal_log.txt", "a")

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

def detect_spike(df, multiplier=2.2):
    avg_body = df.iloc[-20:-1].apply(lambda row: abs(row['close'] - row['open']), axis=1).mean()
    last_body = abs(df.iloc[-1]['close'] - df.iloc[-1]['open'])
    return last_body > avg_body * multiplier

def analyze_symbol(symbol, timeframe='15m'):
    df = get_data(timeframe, symbol)
    if len(df) < 3:
        return None, None

    df['EMA20'] = ta.ema(df['close'], length=20)
    df['EMA50'] = ta.ema(df['close'], length=50)
    df['rsi'] = ta.rsi(df['close'], length=14)
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
    atr = max(atr, entry * MIN_PERCENT_RISK, MIN_ATR)

    above_ema = candle['close'] > candle['EMA20'] and candle['EMA20'] > candle['EMA50']
    below_ema = candle['close'] < candle['EMA20'] and candle['EMA20'] < candle['EMA50']

    direction = None
    if signal_type == 'bullish_marubozu' or signal_type == 'bullish_engulfing':
        if rsi_val < 65 and df['MACD'].iloc[-1] > df['MACDs'].iloc[-1] and adx_val > ADX_THRESHOLD and above_ema:
            direction = 'Long'
    elif signal_type == 'bearish_marubozu' or signal_type == 'bearish_engulfing':
        if rsi_val > 35 and df['MACD'].iloc[-1] < df['MACDs'].iloc[-1] and adx_val > ADX_THRESHOLD and below_ema:
            direction = 'Short'

    if not direction and symbol == 'BTCUSDT' and detect_spike(df):
        direction = 'SPK'

    if direction == 'SPK':
        msg = f"⚡ BTCUSDT Spike Alert!\nTime: {df['timestamp'].iloc[-1]}\nClose: {entry:.2f}"
        return msg, None

    if direction:
        sl = entry - atr * ATR_MULTIPLIER_SL if direction == 'Long' else entry + atr * ATR_MULTIPLIER_SL
        tp1 = entry + atr * TP1_MULTIPLIER if direction == 'Long' else entry - atr * TP1_MULTIPLIER
        tp2 = entry + atr * TP2_MULTIPLIER if direction == 'Long' else entry - atr * TP2_MULTIPLIER
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
""", None

    return None, "❌ No strong candle"

def analyze_symbol_mtf(symbol):
    a5, reason5 = analyze_symbol(symbol, '5m')
    a15, reason15 = analyze_symbol(symbol, '15m')
    if a5 and a15 and (('BUY' in a5 and 'BUY' in a15) or ('SELL' in a5 and 'SELL' in a15)):
        return a15, None
    return None, reason15 or reason5

def monitor():
    symbols = [
        "BTCUSDT", "ETHUSDT", "DOGEUSDT", "BNBUSDT", "XRPUSDT",
        "RENDERUSDT", "TRUMPUSDT", "FARTCOINUSDT", "XLMUSDT",
        "SHIBUSDT", "ADAUSDT", "NOTUSDT"
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
            send_telegram_message("🤖 ربات فعال است و در حال بررسی سیگنال‌ها می‌باشد")
            last_heartbeat = time.time()

        all_reasons = []
        for sym in symbols:
            try:
                msg, reason = analyze_symbol_mtf(sym)
                if msg:
                    send_telegram_message(msg)
                elif reason:
                    all_reasons.append(f"❌ {sym}: {reason}")
            except Exception as e:
                logging.error(f"Error analyzing {sym}: {e}")
        if all_reasons:
            send_telegram_message("📡 No Signals in This Cycle\n" + "\n".join(all_reasons))

        time.sleep(CHECK_INTERVAL)

@app.route('/')
def home():
    return "I'm alive!"

if __name__ == '__main__':
    threading.Thread(target=monitor, daemon=True).start()
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
