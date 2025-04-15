import os
import requests
import time
import logging
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
MONITOR_INTERVAL = 120
SLEEP_HOURS = (0, 7)
MIN_ATR = 0.001
SIGNAL_COOLDOWN = 1800

last_signals = {}
daily_signal_count = 0
daily_hit_count = 0
last_report_day = None
open_positions = {}
tp1_count = 0
tp2_count = 0
sl_count = 0

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ارسال پیام به تلگرام
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            logging.error(f"Telegram error: {response.text}")
    except Exception as e:
        logging.error(f"Telegram exception: {e}")

# دریافت داده‌ها از API CryptoCompare
def get_data(timeframe, symbol):
    url = "https://min-api.cryptocompare.com/data/v2/histominute"
    aggregate = 5 if timeframe == '5m' else 15 if timeframe == '15m' else 1440  # فقط فریم‌های 5m و 15m مجاز هستند
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
    
    if res.status_code == 200:
        data = res.json().get('Data', {}).get('Data', [])
        if not data:
            logging.error(f"No data returned for {symbol} with timeframe {timeframe}")
            return None
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df['volume'] = df['volumeto']
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].dropna()  # حذف داده‌های خالی
        return df
    else:
        logging.error(f"Failed to fetch data for {symbol} with status code {res.status_code}")
        return None

# تحلیل سیگنال‌ها و تعیین الگوهای کندلی
def analyze_symbol(symbol, timeframe='15m'):
    global daily_signal_count

    df = get_data(timeframe, symbol)
    if df is None or len(df) < 30:
        logging.error(f"Not enough data to analyze {symbol} with timeframe {timeframe}")
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
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'])

    candle = df.iloc[-2]
    signal_type = "bullish" if candle['close'] > candle['open'] else "bearish"  # ساده‌ترین تشخیص کندل
    pattern = signal_type.replace("_", " ").title() if signal_type else "None"

    rsi_val = df['rsi'].iloc[-2]
    adx_val = df['ADX'].iloc[-2]
    entry = df['close'].iloc[-2]
    atr = df['ATR'].iloc[-2]
    atr = max(atr, entry * MIN_PERCENT_RISK, MIN_ATR)

    above_ema = candle['close'] > candle['EMA20'] and candle['EMA20'] > candle['EMA50']
    below_ema = candle['close'] < candle['EMA20'] and candle['EMA20'] < candle['EMA50']

    confirmations = []
    if (signal_type == "bullish" and rsi_val >= 50) or (signal_type == "bearish" and rsi_val <= 50):
        confirmations.append("RSI")
    if df['MACD'].iloc[-2] > df['MACDs'].iloc[-2]:
        confirmations.append("MACD")
    if adx_val > ADX_THRESHOLD:
        confirmations.append("ADX")
    if (signal_type == "bullish" and above_ema) or (signal_type == "bearish" and below_ema):
        confirmations.append("EMA")

    confidence = len(confirmations)
    direction = 'Long' if signal_type == "bullish" and confidence >= 3 else 'Short' if signal_type == "bearish" and confidence >= 3 else None

    if direction:
        sl, tp1, tp2 = set_dynamic_stop_loss_take_profit(entry, atr, direction)  # استفاده از SL و TP داینامیک

        message = f"""🚨 *AI Signal Alert*
*Symbol:* `{symbol}`
*Signal:* {'🟢 BUY MARKET' if direction == 'Long' else '🔴 SELL MARKET'}
*Pattern:* {pattern}
*Confidence:* {'🔥' * confidence}
*Entry:* {entry}
*Stop Loss:* {sl}
*Take Profit 1:* {tp1}
*Take Profit 2:* {tp2}"""

        send_telegram_message(message)

    return direction, message

def check_cooldown(symbol, direction):
    # این تابع باید بررسی کند که آیا سیگنال برای این سمبل و جهت ارسال شده است یا نه
    # برای مثال می‌توانید از یک دیکشنری برای ذخیره زمان آخرین سیگنال استفاده کنید
    if symbol in last_signals and last_signals[symbol]['direction'] == direction:
        last_time = last_signals[symbol]['time']
        if time.time() - last_time < SIGNAL_COOLDOWN:
            return True  # سیگنال تکراری است
    last_signals[symbol] = {'direction': direction, 'time': time.time()}
    return False

def analyze_symbol_mtf(symbol):
    # تحلیل سیگنال‌های تایم فریم‌های مختلف
    msg_5m, _ = analyze_symbol(symbol, '5m')
    msg_15m, _ = analyze_symbol(symbol, '15m')
    
    # بررسی سیگنال‌های تکراری
    if msg_5m and msg_15m:
        if ("BUY" in msg_5m and "BUY" in msg_15m) or ("SELL" in msg_5m and "SELL" in msg_15m):
            return msg_15m, None
    elif msg_15m and ("🔥🔥🔥" in msg_15m):
        return msg_15m + "\n⚠️ *Strong 15m signal without 5m confirmation.*", None
    return None, None

def monitor():
    global daily_signal_count, daily_hit_count, last_report_day

    symbols = [
        "BTCUSDT", "ETHUSDT", "DOGEUSDT", "BNBUSDT", "XRPUSDT",
        "RENDERUSDT", "TRUMPUSDT", "FARTCOINUSDT", "XLMUSDT",
        "SHIBUSDT", "ADAUSDT", "NOTUSDT", "PROMUSDT"
    ]
    last_heartbeat = 0

    while True:
        now = datetime.utcnow()
        tehran_hour = (now.hour + 3) % 24
        tehran_min = now.minute
        current_day = now.date()

        # مدت زمان استراحت
        if SLEEP_HOURS[0] <= tehran_hour < SLEEP_HOURS[1]:
            logging.info("Sleeping hours")
            time.sleep(60)
            continue

        if time.time() - last_heartbeat > HEARTBEAT_INTERVAL:
            send_telegram_message("🤖 Bot is alive and scanning signals.")
            last_heartbeat = time.time()

        for sym in symbols:
            try:
                msg, _ = analyze_symbol_mtf(sym)
                if msg:
                    send_telegram_message(msg)
                    daily_hit_count += 1
            except Exception as e:
                logging.error(f"Error analyzing {sym}: {e}")

        time.sleep(CHECK_INTERVAL)

@app.route('/')
def home():
    return "✅ Crypto Signal Bot is running."

if __name__ == '__main__':
    threading.Thread(target=monitor, daemon=True).start()
    threading.Thread(target=monitor_positions, daemon=True).start()
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
