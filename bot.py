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
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

# -------------------------------
# تنظیمات کلی
# -------------------------------
NUM_CANDLES = 60            # تعداد کندل‌های مورد استفاده (داده‌های 15 دقیقه‌ای)
VOLUME_MULTIPLIER = 1.2     # ضریب حجم
PRICE_CHANGE_THRESHOLD = 0.8  # تغییر درصدی قیمت مورد نیاز
STD_MULTIPLIER = 1.0        # ضریب انحراف معیار
ALERT_COOLDOWN = 900        # فاصله زمانی بین هشدارها (15 دقیقه)
HEARTBEAT_INTERVAL = 3600   # پیام هارت‌بییت (۱ ساعت)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

last_alert_time = 0
last_heartbeat_time = 0

# =============================================================================
# بخش اول: توابع مرتبط با API بایننس
# =============================================================================

def get_binance_klines(symbol="BTCUSDT", interval="15m", limit=60):
    """
    دریافت کندل‌های بایننس در تایم‌فریم و تعداد دلخواه.
    interval می‌تواند یکی از مقادیر: 1m, 3m, 5m, 15m, 30m, 1h, 4h, 1d و ...
    """
    base_url = 'https://api.binance.com'
    endpoint = '/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    response = requests.get(base_url + endpoint, params=params, timeout=10)
    data = response.json()  # لیستی از لیست‌ها

    # ساخت DataFrame
    # فرمت هر کندل در بایننس:
    # [
    #   1499040000000,      // open time (ms)
    #   "0.01634790",       // open
    #   "0.80000000",       // high
    #   "0.01575800",       // low
    #   "0.01577100",       // close
    #   "148976.11427815",  // volume
    #   1499644799999,      // close time
    #   "2434.19055334",    // quote asset volume
    #   308,                // number of trades
    #   "1756.87402397",    // taker buy base asset volume
    #   "28.46694368",      // taker buy quote asset volume
    #   "17928899.62484339" // ignore
    # ]

    df = pd.DataFrame(data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_av', 'trades', 'tb_base_av',
        'tb_quote_av', 'ignore'
    ])

    # تبدیل انواع داده
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)

    # فقط ستون‌های اصلی را برمی‌داریم
    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
    return df

# =============================================================================
# بخش دوم: توابع کمکی تحلیل تکنیکال
# =============================================================================

def send_telegram_message(message):
    """ ارسال پیام به تلگرام """
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


def detect_rsi_divergence(df, rsi_period=14, pivot_size=3):
    """
    تشخیص واگرایی RSI با استفاده از شناسایی قله‌ها و دره‌ها در یک پنجره از کندل‌ها.
    در این نسخه، سخت‌گیرانه‌تر عمل می‌کنیم:
    - window_size را 20 می‌گذاریم.
    - pivot_size=3 تا قله/دره‌های واضح‌تر شناسایی شوند.
    """
    try:
        # محاسبه RSI
        df['rsi'] = ta.rsi(df['close'], length=rsi_period)

        # به جای 10 کندل، 20 کندل آخر را بررسی می‌کنیم
        window_size = 20
        if len(df) < window_size:
            return None

        df_window = df.iloc[-window_size:].reset_index(drop=True)
        
        # توابع کمکی برای یافتن قله‌ها و دره‌ها
        def find_peaks(series, left, right):
            peaks = []
            for i in range(left, len(series) - right):
                window = series[i-left:i+right+1]
                if series[i] == max(window):
                    peaks.append(i)
            return peaks

        def find_valleys(series, left, right):
            valleys = []
            for i in range(left, len(series) - right):
                window = series[i-left:i+right+1]
                if series[i] == min(window):
                    valleys.append(i)
            return valleys

        price_peaks = find_peaks(df_window['close'].tolist(), pivot_size, pivot_size)
        price_valleys = find_valleys(df_window['close'].tolist(), pivot_size, pivot_size)
        rsi_peaks = find_peaks(df_window['rsi'].tolist(), pivot_size, pivot_size)
        rsi_valleys = find_valleys(df_window['rsi'].tolist(), pivot_size, pivot_size)

        # بررسی آخرین دو قله برای واگرایی نزولی
        if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
            last_price_peak = price_peaks[-1]
            prev_price_peak = price_peaks[-2]
            if df_window['close'].iloc[last_price_peak] > df_window['close'].iloc[prev_price_peak]:
                last_rsi_peak = rsi_peaks[-1]
                prev_rsi_peak = rsi_peaks[-2]
                if df_window['rsi'].iloc[last_rsi_peak] < df_window['rsi'].iloc[prev_rsi_peak]:
                    return "واگرایی نزولی (Bearish Divergence)"

        # بررسی آخرین دو دره برای واگرایی صعودی
        if len(price_valleys) >= 2 and len(rsi_valleys) >= 2:
            last_price_valley = price_valleys[-1]
            prev_price_valley = price_valleys[-2]
            if df_window['close'].iloc[last_price_valley] < df_window['close'].iloc[prev_price_valley]:
                last_rsi_valley = rsi_valleys[-1]
                prev_rsi_valley = rsi_valleys[-2]
                if df_window['rsi'].iloc[last_rsi_valley] > df_window['rsi'].iloc[prev_rsi_valley]:
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
    volumes = [candle.get('volume', 0) for candle in candles[:-1]]
    return mean(volumes) * VOLUME_MULTIPLIER

def calculate_price_spike(candles):
    close_prices = [candle['close'] for candle in candles[:-1]]
    if len(close_prices) < 2:
        return 0, None
    price_changes = []
    for i in range(1, len(close_prices)):
        change = (close_prices[i] - close_prices[i - 1]) / close_prices[i - 1] * 100
        price_changes.append(change)
    avg_change = mean(price_changes)
    try:
        change_std = stdev(price_changes)
    except:
        change_std = 0
    previous_close = candles[-2]['close']
    current_close = candles[-1]['close']
    current_change = (current_close - previous_close) / previous_close * 100
    spike_type = None
    if current_change >= PRICE_CHANGE_THRESHOLD and (current_change - avg_change >= STD_MULTIPLIER * change_std):
        spike_type = 'UP'
    elif current_change <= -PRICE_CHANGE_THRESHOLD and (avg_change - current_change >= STD_MULTIPLIER * change_std):
        spike_type = 'DOWN'
    return current_change, spike_type

def check_spike(candles):
    if len(candles) < NUM_CANDLES + 1:
        logging.warning("تعداد کندل‌ها کمتر از حد مورد نیاز است.")
        return None, 0
    current_volume = candles[-1].get('volume', 0)
    volume_threshold = calculate_volume_threshold(candles)
    volume_spike = current_volume > volume_threshold
    current_price_change, spike_type = calculate_price_spike(candles)
    if volume_spike and spike_type is not None:
        return spike_type, current_price_change
    if not volume_spike:
        logging.info(f"حجم کافی نبود. حجم: {current_volume:.2f}, آستانه: {volume_threshold:.2f}")
    if spike_type is None:
        logging.info(f"تغییر قیمت ({current_price_change:.2f}%) در محدوده جهش نبود یا انحراف معیار کافی نبود.")
    return None, current_price_change

def is_big_green_candle(row, threshold=2.0):
    try:
        if row['open'] == 0:
            return False
        body_pct = (row['close'] - row['open']) / row['open'] * 100
        return body_pct >= threshold
    except Exception as e:
        logging.error(f"خطا در is_big_green_candle: {e}")
        return False

def is_price_rise_above_threshold(df, threshold=2.0):
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

# =============================================================================
# بخش سوم: نظارت بر BTC (15 دقیقه) و ارسال سیگنال Spike
# =============================================================================

def get_bitcoin_data():
    try:
        df = get_binance_klines(symbol="BTCUSDT", interval="15m", limit=NUM_CANDLES)
        df.rename(columns={'open_time': 'timestamp'}, inplace=True)
        return df
    except Exception as e:
        logging.error("خطا در get_bitcoin_data: " + str(e))
        return pd.DataFrame()

def monitor_bitcoin():
    global last_alert_time, last_heartbeat_time
    logging.info("شروع نظارت بر BTC/USDT (15m)...")
    send_telegram_message("سیستم نظارت BTC/USDT فعال شد (کندل‌های 15 دقیقه‌ای - منبع بایننس).")
    last_heartbeat_time = time.time()
    while True:
        try:
            df = get_bitcoin_data()
            if df.empty or len(df) < 3:
                logging.info("داده‌های BTC/USDT کافی نیستند.")
            else:
                # تبدیل df به candles (لیست دیکشنری)
                candles = df.to_dict(orient="records")
                spike_type, price_change = check_spike(candles)
                if spike_type is not None:
                    current_time = time.time()
                    if (current_time - last_alert_time) >= ALERT_COOLDOWN:
                        if spike_type == 'UP':
                            message = (
                                f"📈 جهش صعودی BTC/USDT تشخیص داده شد!\n"
                                f"تغییر قیمت: {price_change:.2f}%\n"
                                f"حجم: {candles[-1].get('volume', 'N/A')}"
                            )
                        else:
                            message = (
                                f"📉 جهش نزولی BTC/USDT تشخیص داده شد!\n"
                                f"تغییر قیمت: {price_change:.2f}%\n"
                                f"حجم: {candles[-1].get('volume', 'N/A')}"
                            )
                        send_telegram_message(message)
                        logging.info(message)
                        last_alert_time = current_time
                    else:
                        logging.info("سیگنال BTC/USDT یافت شد ولی دوره‌ی Cooldown فعال است.")
                else:
                    logging.info(f"هیچ سیگنال BTC/USDT یافت نشد. تغییر قیمت: {price_change:.2f}%")
            
            # پیام Heartbeat هر ۱ ساعت
            if time.time() - last_heartbeat_time >= HEARTBEAT_INTERVAL:
                send_telegram_message("سیستم نظارت BTC/USDT همچنان فعال است (منبع بایننس).")
                last_heartbeat_time = time.time()

            logging.info("چرخه نظارت BTC/USDT تکمیل شد.")
            time.sleep(900)  # بررسی هر 15 دقیقه
        except Exception as ex:
            logging.error("خطای غیرمنتظره در monitor_bitcoin: " + str(ex))
            time.sleep(60)

# =============================================================================
# بخش چهارم: تحلیل چند ارز (15 دقیقه) و ارسال سیگنال تکنیکال
# =============================================================================

def get_symbol_data(symbol="BTCUSDT", interval="15m", limit=60):
    """
    دریافت داده‌ی نماد از بایننس و تبدیل به DataFrame سازگار با توابع تحلیل.
    """
    try:
        df = get_binance_klines(symbol=symbol, interval=interval, limit=limit)
        df.rename(columns={'open_time': 'timestamp'}, inplace=True)
        return df
    except Exception as e:
        logging.error(f"خطا در get_symbol_data برای {symbol}: {e}")
        return pd.DataFrame()

def analyze_symbol(symbol="BTCUSDT"):
    df = get_symbol_data(symbol, interval="15m", limit=60)
    if df.empty or len(df) < 3:
        return f"تحلیل بازار برای {symbol}: داده‌های کافی دریافت نشد."

    # اعمال توابع تحلیل
    df = find_support_resistance(df)
    trend = find_trendline(df)
    divergence = detect_rsi_divergence(df)
    rsi_val = df['rsi'].iloc[-1] if 'rsi' in df.columns else None
    pin_bar = df.apply(is_pin_bar, axis=1).iloc[-1]
    doji = df.apply(is_doji, axis=1).iloc[-1]
    big_green = df.apply(is_big_green_candle, axis=1).iloc[-1]
    price_rise_2pct = is_price_rise_above_threshold(df, 2.0)

    # تعیین سیگنال
    signal = "سیگنالی یافت نشد"
    if pin_bar and rsi_val is not None and rsi_val < 30:
        signal = "ورود به پوزیشن Long (Pin Bar + RSI زیر 30)"
    elif pin_bar and rsi_val is not None and rsi_val > 70:
        signal = "ورود به پوزیشن Short (Pin Bar + RSI بالای 70)"
    elif doji:
        signal = "الگوی دوجی شناسایی شد"
    elif divergence:
        signal = f"واگرایی شناسایی شد: {divergence}"
    elif big_green:
        signal = "کندل صعودی قدرتمند شناسایی شد (Big Green Candle)"
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
    # مثال نمادها (بدون اسلش): BTCUSDT, ETHUSDT, SHIBUSDT ...
    symbols = [
        'BTCUSDT', 'ETHUSDT', 'SHIBUSDT', 'NEARUSDT',
        'SOLUSDT', 'DOGEUSDT', 'MATICUSDT', 'BNBUSDT',
        'WIFUSDT', 'VIRTUALUSDT', 'ENAUSDT'
    ]
    while True:
        try:
            for symbol in symbols:
                logging.info(f"در حال بررسی {symbol}...")
                try:
                    analysis_message = analyze_symbol(symbol)
                    logging.info(f"نتیجه تحلیل {symbol}: {analysis_message.strip()}")
                    # اگر سیگنالی یافت شد (عبارت "سیگنالی یافت نشد" در پیام نبود)، ارسال به تلگرام
                    if "سیگنال:" in analysis_message and "سیگنالی یافت نشد" not in analysis_message:
                        send_telegram_message(analysis_message)
                except Exception as e:
                    logging.error(f"خطا در بررسی {symbol}: {e}")

            logging.info("چرخه تحلیل چند ارز تکمیل شد.")
            time.sleep(900)  # بررسی هر 15 دقیقه
        except Exception as ex:
            logging.error("خطای غیرمنتظره در multi_symbol_analysis_loop: " + str(ex))
            time.sleep(60)

# =============================================================================
# بخش پنجم: اجرای دو سیستم به صورت همزمان + Flask
# =============================================================================

def run_all_systems():
    btc_thread = threading.Thread(target=monitor_bitcoin, daemon=True)
    multi_thread = threading.Thread(target=multi_symbol_analysis_loop, daemon=True)
    btc_thread.start()
    multi_thread.start()

@app.route('/')
def home():
    return "I'm alive!"

if __name__ == '__main__':
    from threading import Thread
    Thread(target=run_all_systems, daemon=True).start()
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
