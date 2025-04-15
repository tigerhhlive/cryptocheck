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
    
    # تغییرات فریم‌های زمانی بلندمدت
    aggregate = 5 if timeframe == '5m' else 15 if timeframe == '15m' else 30 if timeframe == '30m' else 60 if timeframe == '1h' else 1440  # روزانه (1d)
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
    
    # تغییرات: تبدیل 'time' به 'timestamp' و 'volumeto' به 'volume'
    df['timestamp'] = pd.to_datetime(df['time'], unit='s')  # تبدیل 'time' به 'timestamp'
    df['volume'] = df['volumeto']  # 'volumeto' به 'volume' تبدیل می‌شود
    
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

# اضافه کردن API اخبار
def fetch_news():
    url = "https://cryptocontrol.io/api/v1/public/news"
    headers = {
        'Authorization': 'Bearer 3788a1f05c7d472a94700d5c35cd465f'  # API Key به‌طور مستقیم در هدر
    }
    params = {
        'lang': 'en',  # زبان اخبار انگلیسی
        'categories': 'all',  # دسته‌بندی اخبار
        'limit': 5  # محدود به 5 خبر
    }
    
    # ارسال درخواست به API
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # بررسی وضعیت پاسخ (برای شبیه‌سازی خطا در صورت لزوم)
        
        news_data = response.json()  # تبدیل داده‌های JSON به دیکشنری
        
        if 'data' in news_data:  # بررسی وجود داده‌ها
            return news_data['data']  # داده‌های خبری را برمی‌گرداند
        else:
            print("No news data found.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return None

def analyze_sentiment():
    """
    این تابع اخبار بازار را دریافت کرده و تحلیل احساسات بازار را انجام می‌دهد.
    """
    news_data = fetch_news()
    
    # تحلیل اخبار (میتوانید از مدل‌های پیشرفته‌تر برای تحلیل احساسات استفاده کنید)
    sentiment_score = 0
    for news_item in news_data['data']:
        sentiment_score += int(news_item['positive'] - news_item['negative'])  # تحلیل ساده برای احساسات

    return sentiment_score

def set_dynamic_stop_loss_take_profit(entry, atr, direction):
    """
    این تابع برای تنظیم داینامیک SL و TP برای معامله استفاده می‌شود.
    """
    if direction == 'Long':
        sl = entry - atr * ATR_MULTIPLIER_SL
        tp1 = entry + atr * TP1_MULTIPLIER
        tp2 = entry + atr * TP2_MULTIPLIER
    elif direction == 'Short':
        sl = entry + atr * ATR_MULTIPLIER_SL
        tp1 = entry - atr * TP1_MULTIPLIER
        tp2 = entry - atr * TP2_MULTIPLIER

    return sl, tp1, tp2

def dynamic_threshold_adjustment(df):
    """
    این تابع برای تنظیم آستانه‌های هوشمند (مثل RSI و EMA) قبل از محاسبه اندیکاتورها استفاده می‌شود.
    """
    # تنظیمات هوشمند برای اندیکاتورها (می‌توانید این را بر اساس نیاز خود تنظیم کنید)
    if df['rsi'].iloc[-2] < 30:
        rsi_threshold = 25  # تنظیم آستانه برای RSI پایین
    elif df['rsi'].iloc[-2] > 70:
        rsi_threshold = 75  # تنظیم آستانه برای RSI بالا
    else:
        rsi_threshold = 50  # آستانه معمولی برای RSI

    # برای EMA و دیگر اندیکاتورها هم می‌توان همین کار را انجام داد.
    return rsi_threshold

def analyze_symbol(symbol, timeframe='15m'):
    global daily_signal_count

    df = get_data(timeframe, symbol)
    if len(df) < 30:
        return None, None

    rsi_threshold = dynamic_threshold_adjustment(df)  # تنظیم آستانه‌های هوشمند

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
    signal_type = detect_strong_candle(candle) or detect_engulfing(df[:-1])
    pattern = signal_type.replace("_", " ").title() if signal_type else "None"

    rsi_val = df['rsi'].iloc[-2]
    adx_val = df['ADX'].iloc[-2]
    entry = df['close'].iloc[-2]
    atr = df['ATR'].iloc[-2]
    atr = max(atr, entry * MIN_PERCENT_RISK, MIN_ATR)

    # استفاده از تحلیل احساسات بازار
    sentiment_score = analyze_sentiment()

    above_ema = candle['close'] > candle['EMA20'] and candle['EMA20'] > candle['EMA50']
    below_ema = candle['close'] < candle['EMA20'] and candle['EMA20'] < candle['EMA50']

    confirmations = []
    if (signal_type and 'bullish' in signal_type and rsi_val >= 50) or (signal_type and 'bearish' in signal_type and rsi_val <= 50):
        confirmations.append("RSI")
    if (df['MACD'].iloc[-2] > df['MACDs'].iloc[-2]) if 'bullish' in str(signal_type) else (df['MACD'].iloc[-2] < df['MACDs'].iloc[-2]):
        confirmations.append("MACD")
    if adx_val > ADX_THRESHOLD:
        confirmations.append("ADX")
    if ('bullish' in str(signal_type) and above_ema) or ('bearish' in str(signal_type) and below_ema):
        confirmations.append("EMA")

    confidence = len(confirmations)
    direction = 'Long' if 'bullish' in str(signal_type) and confidence >= 3 else 'Short' if 'bearish' in str(signal_type) and confidence >= 3 else None

    if direction and not check_cooldown(symbol, direction):
        logging.info(f"{symbol} - DUPLICATE SIGNAL - Skipped due to cooldown")
        return None, "Duplicate"

    if direction:
        daily_signal_count += 1
        sl, tp1, tp2 = set_dynamic_stop_loss_take_profit(entry, atr, direction)  # استفاده از SL و TP داینامیک
        confidence_stars = "🔥" * confidence

        message = f"""🚨 *AI Signal Alert*
*Symbol:* `{symbol}`
*Signal:* {'🟢 BUY MARKET' if direction == 'Long' else '🔴 SELL MARKET'}
*Pattern:* {pattern}
*Confidence:* {confidence_stars}
*Entry:* {entry}
*Stop Loss:* {sl}
*Take Profit 1:* {tp1}
*Take Profit 2:* {tp2}"""

        send_telegram_message(message)

    return direction, message


def analyze_symbol_mtf(symbol):
    msg_5m, _ = analyze_symbol(symbol, '5m')
    msg_15m, _ = analyze_symbol(symbol, '15m')
    if msg_5m and msg_15m:
        if ("BUY" in msg_5m and "BUY" in msg_15m) or ("SELL" in msg_5m and "SELL" in msg_15m):
            return msg_15m, None
    elif msg_15m and ("🔥🔥🔥" in msg_15m):
        return msg_15m + "\n⚠️ *Strong 15m signal without 5m confirmation.*", None
    return None, None

def monitor_positions():
    """
    تابع برای بررسی وضعیت پوزیشن‌ها و مدیریت آن‌ها.
    """
    # کد مدیریت پوزیشن‌ها (فرضی یا می‌تواند بسته به نیاز شما اضافه شود)
    while True:
        # بررسی وضعیت پوزیشن‌ها
        # و سایر وظایف مرتبط
        time.sleep(MONITOR_INTERVAL)

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
