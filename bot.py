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

# -------------------------------
# تنظیمات محیطی (Secrets)
# -------------------------------
CRYPTOCOMPARE_API_KEY = os.environ.get('CRYPTOCOMPARE_API_KEY')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

# -------------------------------
# تنظیمات تحلیل برای BTC/USDT
# -------------------------------
NUM_CANDLES = 60           # تعداد کندل‌های مورد استفاده (داده‌های 5 دقیقه‌ای)
VOLUME_MULTIPLIER = 1.2    # ضریب حجم: کمی کمتر سخت‌گیرانه (قبلا 1.5 بود)
PRICE_CHANGE_THRESHOLD = 0.8  # تغییر درصدی قیمت مورد نیاز (کمی کاهش دادیم تا جهش راحت‌تر شناسایی شود)
STD_MULTIPLIER = 1.0       # ضریب انحراف معیار (قبلا 1.5 بود)
ALERT_COOLDOWN = 300       # فاصله زمانی بین هشدارها (۵ دقیقه)
HEARTBEAT_INTERVAL = 3600  # پیام هارت‌بییت (۱ ساعت)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

last_alert_time = 0
last_heartbeat_time = 0


# -------------------------------
# تابع ارسال پیام به تلگرام
# -------------------------------
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            logging.info("پیام به تلگرام ارسال شد: " + message)
        else:
            logging.error(f"خطا در ارسال پیام تلگرام: {response.text}")
    except Exception as e:
        logging.error(f"خطا در ارسال پیام تلگرام: {e}")


# -------------------------------
# توابع تحلیل تکنیکال مشترک
# -------------------------------
def find_support_resistance(df, window=5):
    try:
        df['support'] = df['low'].rolling(window=window, center=False).min()
        df['resistance'] = df['high'].rolling(window=window, center=False).max()
        return df
    except Exception as e:
        logging.error("خطا در find_support_resistance: " + str(e))
        return df


def find_trendline(df):
    try:
        if len(df) < 3:
            return "روند خنثی"
        if df['close'].iloc[-1] > df['close'].iloc[-2] > df['close'].iloc[-3]:
            return "روند صعودی"
        elif df['close'].iloc[-1] < df['close'].iloc[-2] < df['close'].iloc[-3]:
            return "روند نزولی"
        return "روند خنثی"
    except Exception as e:
        logging.error("خطا در find_trendline: " + str(e))
        return "روند خنثی"


def detect_rsi_divergence(df, rsi_period=14):
    try:
        df['rsi'] = ta.rsi(df['close'], length=rsi_period)
        if len(df) < 2:
            return None
        # واگرایی نزولی
        if (df['close'].iloc[-1] > df['close'].iloc[-2]) and (df['rsi'].iloc[-1] < df['rsi'].iloc[-2]):
            return "واگرایی نزولی (Bearish Divergence)"
        # واگرایی صعودی
        elif (df['close'].iloc[-1] < df['close'].iloc[-2]) and (df['rsi'].iloc[-1] > df['rsi'].iloc[-2]):
            return "واگرایی صعودی (Bullish Divergence)"
        return None
    except Exception as e:
        logging.error("خطا در detect_rsi_divergence: " + str(e))
        return None


def is_pin_bar(row):
    try:
        body_size = abs(row['close'] - row['open'])
        upper_shadow = row['high'] - max(row['close'], row['open'])
        lower_shadow = min(row['close'], row['open']) - row['low']
        return body_size < lower_shadow and body_size < upper_shadow
    except Exception as e:
        logging.error("خطا در is_pin_bar: " + str(e))
        return False


def is_doji(row):
    try:
        body_size = abs(row['close'] - row['open'])
        candle_range = row['high'] - row['low']
        if candle_range == 0:
            return False
        return body_size <= 0.1 * candle_range
    except Exception as e:
        logging.error("خطا در is_doji: " + str(e))
        return False


# -------------------------------
# توابع تشخیص جهش (Spike)
# -------------------------------
def calculate_volume_threshold(candles):
    # میانگین حجم کندل‌های قبلی
    volumes = [candle.get('volumefrom', candle.get('volume', 0)) for candle in candles[:-1]]
    return mean(volumes) * VOLUME_MULTIPLIER

def calculate_price_spike(candles):
    close_prices = [candle['close'] for candle in candles[:-1]]
    if len(close_prices) < 2:
        return 0, None

    # محاسبه درصد تغییر قیمت کندل‌های قبلی
    price_changes = []
    for i in range(1, len(close_prices)):
        change = (close_prices[i] - close_prices[i - 1]) / close_prices[i - 1] * 100
        price_changes.append(change)

    avg_change = mean(price_changes)
    try:
        change_std = stdev(price_changes)
    except:
        change_std = 0

    # درصد تغییر کندل فعلی نسبت به کندل قبلی
    previous_close = candles[-2]['close']
    current_close = candles[-1]['close']
    current_change = (current_close - previous_close) / previous_close * 100

    spike_type = None
    # شرط صعودی
    if current_change >= PRICE_CHANGE_THRESHOLD and (current_change - avg_change >= STD_MULTIPLIER * change_std):
        spike_type = 'UP'
    # شرط نزولی
    elif current_change <= -PRICE_CHANGE_THRESHOLD and (avg_change - current_change >= STD_MULTIPLIER * change_std):
        spike_type = 'DOWN'

    return current_change, spike_type


def check_spike(candles):
    if len(candles) < NUM_CANDLES + 1:
        logging.warning("تعداد کندل‌ها کمتر از حد مورد نیاز است.")
        return None, 0

    current_volume = candles[-1].get('volumefrom', candles[-1].get('volume', 0))
    volume_threshold = calculate_volume_threshold(candles)
    volume_spike = current_volume > volume_threshold

    current_price_change, spike_type = calculate_price_spike(candles)

    if volume_spike and spike_type is not None:
        return spike_type, current_price_change

    # اگر سیگنال صادر نشده، در لاگ ذکر کنیم
    if not volume_spike:
        logging.info(f"حجم برای جهش کافی نبود. حجم فعلی: {current_volume:.2f}, آستانه: {volume_threshold:.2f}")
    if spike_type is None:
        logging.info(f"تغییر قیمت ({current_price_change:.2f}%) در محدوده‌ی جهش نبود یا انحراف معیار کافی نبوده است.")

    return None, current_price_change


# -------------------------------
# توابع جدید برای تشخیص کندل‌های قدرتمند
# -------------------------------
def is_big_green_candle(row, threshold=2.0):
    """
    تشخیص کندل سبز قدرتمند: اگر درصد افزایش قیمت از قیمت باز حداقل threshold درصد باشد.
    """
    try:
        if row['open'] == 0:
            return False
        body_pct = (row['close'] - row['open']) / row['open'] * 100
        return body_pct >= threshold
    except Exception as e:
        logging.error(f"خطا در is_big_green_candle: {e}")
        return False


def is_price_rise_above_threshold(df, threshold=2.0):
    """
    تشخیص افزایش قیمت بیش از threshold درصد نسبت به کندل قبلی.
    """
    try:
        if len(df) < 2:
            return False
        prev_close = df['close'].iloc[-2]
        current_close = df['close'].iloc[-1]
        if prev_close == 0:
            return False
        change_pct = (current_close - prev_close) / prev_close * 100
        return change_pct >= threshold
    except Exception as e:
        logging.error(f"خطا در is_price_rise_above_threshold: {e}")
        return False


# -------------------------------
# بخش اول: نظارت BTC/USDT (سیستم مجزا)
# -------------------------------
def get_bitcoin_data():
    url = "https://min-api.cryptocompare.com/data/v2/histominute"
    params = {
        'fsym': 'BTC',
        'tsym': 'USDT',
        'limit': NUM_CANDLES,
        'aggregate': 5,  # هر کندل = 5 دقیقه
        'e': 'CCCAGG',
        'api_key': CRYPTOCOMPARE_API_KEY
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data_json = response.json()
        if data_json.get('Response') != 'Success':
            raise ValueError("خطا در دریافت داده‌ها: " + data_json.get('Message', 'Unknown error'))
        data = data_json['Data']['Data']
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volumeto': 'volume'
        }, inplace=True)
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        return df
    except Exception as e:
        logging.error("خطا در get_bitcoin_data: " + str(e))
        return pd.DataFrame()


def monitor_bitcoin():
    global last_alert_time, last_heartbeat_time
    logging.info("شروع نظارت بر BTC/USDT...")
    send_telegram_message("سیستم نظارت BTC/USDT فعال شد (کندل‌های 5 دقیقه‌ای، شاخص CCCAGG).")
    last_heartbeat_time = time.time()

    while True:
        try:
            df = get_bitcoin_data()
            if df.empty or len(df) < 3:
                logging.info("داده‌های BTC/USDT کافی نیستند.")
            else:
                candles = df.to_dict(orient="records")
                spike_type, price_change = check_spike(candles)

                if spike_type is not None:
                    current_time = time.time()
                    if (current_time - last_alert_time) >= ALERT_COOLDOWN:
                        if spike_type == 'UP':
                            message = (
                                f"📈 جهش صعودی BTC/USDT تشخیص داده شد!\n"
                                f"تغییر قیمت: {price_change:.2f}%\n"
                                f"حجم: {candles[-1].get('volumefrom', candles[-1].get('volume', 'N/A'))}"
                            )
                        else:
                            message = (
                                f"📉 جهش نزولی BTC/USDT تشخیص داده شد!\n"
                                f"تغییر قیمت: {price_change:.2f}%\n"
                                f"حجم: {candles[-1].get('volumefrom', candles[-1].get('volume', 'N/A'))}"
                            )
                        send_telegram_message(message)
                        logging.info(message)
                        last_alert_time = current_time
                    else:
                        logging.info("سیگنال BTC/USDT یافت شد ولی دوره‌ی Cooldown فعال است.")
                else:
                    logging.info(f"هیچ سیگنال BTC/USDT یافت نشد. تغییر قیمت: {price_change:.2f}%")

            # هارت‌بییت
            if time.time() - last_heartbeat_time >= HEARTBEAT_INTERVAL:
                send_telegram_message("سیستم نظارت BTC/USDT همچنان فعال است (CCCAGG).")
                last_heartbeat_time = time.time()

            logging.info("چرخه نظارت BTC/USDT تکمیل شد.")
            time.sleep(300)

        except Exception as ex:
            logging.error("خطای غیرمنتظره در monitor_bitcoin: " + str(ex))
            time.sleep(60)


# -------------------------------
# بخش دوم: تحلیل چند ارز (سیستم مجزا)
# -------------------------------
def get_price_data(symbol, timeframe, limit=100):
    try:
        if timeframe == '1h':
            url = "https://min-api.cryptocompare.com/data/v2/histominute"
            aggregate = 1  # هر 1 دقیقه
            limit = 60     # 60 داده برای 1 ساعت
        elif timeframe == '1d':
            url = "https://min-api.cryptocompare.com/data/v2/histohour"
            aggregate = 1  # هر 1 ساعت
            limit = 24     # 24 داده برای 1 روز
        else:
            raise ValueError("تایم‌فریم پشتیبانی نمی‌شود. فقط '1h' یا '1d' مجاز است.")

        params = {
            'fsym': symbol.split('/')[0],
            'tsym': symbol.split('/')[1],
            'limit': limit,
            'aggregate': aggregate,
            'e': 'CCCAGG'
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()['Data']['Data']
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volumeto': 'volume'
        }, inplace=True)
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        return df
    except Exception as e:
        logging.error(f"خطا در get_price_data برای {symbol}: {e}")
        return pd.DataFrame()


def analyze_symbol(symbol, timeframe='1h'):
    df = get_price_data(symbol, timeframe)
    if df.empty or len(df) < 3:
        return f"تحلیل بازار برای {symbol}: داده‌های کافی دریافت نشد."

    # حمایت/مقاومت و روند
    df = find_support_resistance(df)
    trend = find_trendline(df)

    # RSI و واگرایی
    divergence = detect_rsi_divergence(df)
    rsi_val = df['rsi'].iloc[-1] if 'rsi' in df.columns else None

    # الگوهای کندلی
    pin_bar = df.apply(is_pin_bar, axis=1).iloc[-1]
    doji = df.apply(is_doji, axis=1).iloc[-1]

    # شرایط جدید: کندل سبز قدرتمند و افزایش قیمت بیش از 2٪ نسبت به کندل قبلی
    big_green = df.apply(is_big_green_candle, axis=1).iloc[-1]
    price_rise_2pct = is_price_rise_above_threshold(df, 2.0)

    signal = "سیگنالی یافت نشد"
    # اگر Pin Bar و RSI زیر 30 => سیگنال Long
    if pin_bar and rsi_val is not None and rsi_val < 30:
        signal = "ورود به پوزیشن Long (Pin Bar + RSI زیر 30)"
    # اگر Pin Bar و RSI بالای 70 => سیگنال Short
    elif pin_bar and rsi_val is not None and rsi_val > 70:
        signal = "ورود به پوزیشن Short (Pin Bar + RSI بالای 70)"
    # اگر دوجی => فقط اطلاع‌دهی
    elif doji:
        signal = "الگوی دوجی شناسایی شد"
    # اگر واگرایی داشت => هشدار واگرایی
    elif divergence:
        signal = f"واگرایی شناسایی شد: {divergence}"
    # اگر کندل سبز قدرتمند
    elif big_green:
        signal = "کندل صعودی قدرتمند شناسایی شد (Big Green Candle)"
    # اگر افزایش قیمت بیش از 2٪
    elif price_rise_2pct:
        signal = "افزایش قیمت بیش از ۲٪ در کندل اخیر"

    message = f"""
تحلیل بازار برای {symbol}:
- قیمت فعلی: {df['close'].iloc[-1]}
- حمایت: {df['support'].iloc[-1]}
- مقاومت: {df['resistance'].iloc[-1]}
- خط روند: {trend}
- RSI: {rsi_val}
- سیگنال: {signal}
"""
    return message


def multi_symbol_analysis_loop():
    symbols = [
        'BTC/USDT', 'ETH/USDT', 'NOT/USDT', 'ONDO/USDT', 'WIF/USDT',
        'NEAR/USDT', 'SOL/USDT', 'DOGE/USDT', 'ENA/USDT', 'VIRTUAL/USDT',
        'SHIB/USDT'
    ]
    while True:
        try:
            for symbol in symbols:
                logging.info(f"در حال بررسی {symbol}...")
                try:
                    analysis_message = analyze_symbol(symbol, '1h')
                    logging.info(f"نتیجه تحلیل {symbol}: {analysis_message.strip()}")
                    # اگر در پیام، عبارت "سیگنال:" وجود داشت و "سیگنالی یافت نشد" نبود، پیام به تلگرام ارسال می‌کنیم
                    if "سیگنال:" in analysis_message and "سیگنالی یافت نشد" not in analysis_message:
                        send_telegram_message(analysis_message)
                except Exception as e:
                    logging.error(f"خطا در بررسی {symbol}: {e}")
            logging.info("چرخه تحلیل چند ارز تکمیل شد.")
            time.sleep(300)
        except Exception as ex:
            logging.error("خطای غیرمنتظره در multi_symbol_analysis_loop: " + str(ex))
            time.sleep(60)


# -------------------------------
# اجرای دو سیستم به صورت مجزا
# -------------------------------
def run_all_systems():
    btc_thread = threading.Thread(target=monitor_bitcoin, daemon=True)
    multi_thread = threading.Thread(target=multi_symbol_analysis_loop, daemon=True)
    btc_thread.start()
    multi_thread.start()


# -------------------------------
# روت ساده Flask برای نگهداری سرویس (برای UptimeRobot)
# -------------------------------
@app.route('/')
def home():
    return "I'm alive!"


if __name__ == '__main__':
    from threading import Thread
    # اجرای سیستم‌ها در پس‌زمینه
    Thread(target=run_all_systems, daemon=True).start()
    # اجرای Flask
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
