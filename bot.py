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

def check_cooldown(symbol, direction):
    key = f"{symbol}_{direction}"
    last_time = last_signals.get(key)
    now = time.time()
    if last_time and (now - last_time < SIGNAL_COOLDOWN):
        return False
    last_signals[key] = now
    return True

def analyze_symbol(symbol, timeframe='15m'):
    log_prefix = f"[{datetime.utcnow()}] {symbol} [{timeframe}]"
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

    conditions = {
        "rsi": rsi_val >= 50 if signal_type and 'bullish' in signal_type else rsi_val <= 50,
        "macd": (df['MACD'].iloc[-1] > df['MACDs'].iloc[-1]) if 'bullish' in str(signal_type) else (df['MACD'].iloc[-1] < df['MACDs'].iloc[-1]),
        "adx": adx_val > ADX_THRESHOLD,
        "ema": above_ema if 'bullish' in str(signal_type) else below_ema,
    }

    valid_conditions = sum(1 for v in conditions.values() if v)
    direction = 'Long' if 'bullish' in str(signal_type) and valid_conditions >= 3 else \
                'Short' if 'bearish' in str(signal_type) and valid_conditions >= 3 else None

    reason = [] if direction else [k for k, v in conditions.items() if not v]

    if direction and not check_cooldown(symbol, direction):
        log_file.write(f"{log_prefix} - DUPLICATE SIGNAL - Skipped due to cooldown\n")
        return None, "Duplicate signal cooldown"

    if direction:
        log_file.write(f"{log_prefix} - SIGNAL: {'BUY' if direction == 'Long' else 'SELL'} | Conditions Passed: {valid_conditions}/4\n")
        sl = entry - atr * ATR_MULTIPLIER_SL if direction == 'Long' else entry + atr * ATR_MULTIPLIER_SL
        tp1 = entry + atr * TP1_MULTIPLIER if direction == 'Long' else entry - atr * TP1_MULTIPLIER
        tp2 = entry + atr * TP2_MULTIPLIER if direction == 'Long' else entry - atr * TP2_MULTIPLIER
        rr_ratio = abs(tp1 - entry) / abs(entry - sl)
        TP1_MULT = max(1.5, round(rr_ratio * 1.1, 1))
        TP2_MULT = round(TP1_MULT * 1.5, 1)
        tp1 = entry + atr * TP1_MULT if direction == 'Long' else entry - atr * TP1_MULT
        tp2 = entry + atr * TP2_MULT if direction == 'Long' else entry - atr * TP2_MULT
        message = f"""🚨 *AI Signal Alert*
*Symbol:* `{symbol}`
*Signal:* {'🟢 BUY MARKET' if direction == 'Long' else '🔴 SELL MARKET'}
*Entry:* `{entry:.6f}`
*Stop Loss:* `{sl:.6f}`
*Target 1:* `{tp1:.6f}`
*Target 2:* `{tp2:.6f}`
*Leverage (est.):* `{rr_ratio:.2f}X`"""
        return message, None

    log_file.write(f"{log_prefix} - NO SIGNAL | Conditions Passed: {valid_conditions}/4 | Failed: {', '.join(reason)}\n")
    return None, None

def analyze_symbol_mtf(symbol):
    a5, reason5 = analyze_symbol(symbol, '5m')
    a15, reason15 = analyze_symbol(symbol, '15m')
    if a5 and a15 and (("BUY" in a5 and "BUY" in a15) or ("SELL" in a5 and "SELL" in a15)):
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
                
            except Exception as e:
                logging.error(f"Error analyzing {sym}: {e}")
        time.sleep(CHECK_INTERVAL)

@app.route('/')
def home():
    return "I'm alive!"

@app.route('/logs')
def view_logs():
    try:
        with open("ai_signal_log.txt", "r") as file:
            content = file.read()
        if content:
            return f"<pre>{content}</pre>"
        else:
            return "✅ فایل لاگ هنوز خالی است."
    except Exception as e:
        return f"❌ خطا در خواندن لاگ: {e}"


if __name__ == '__main__':
    threading.Thread(target=monitor, daemon=True).start()
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
