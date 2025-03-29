import os
import time
import logging
import requests
from statistics import mean, stdev
import threading
import pandas as pd
import pandas_ta as ta
from flask import Flask

# -------------------------------
# ساخت Flask App
# -------------------------------
app = Flask(__name__)

# -------------------------------
# تنظیمات محیطی (Secrets)
# -------------------------------
CRYPTOCOMPARE_API_KEY = os.environ.get('CRYPTOCOMPARE_API_KEY')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

# -------------------------------
# تنظیمات کلی
# -------------------------------
NUM_CANDLES = 60            # تعداد کندل‌های مورد استفاده (15 دقیقه‌ای)
VOLUME_MULTIPLIER = 1.2     
PRICE_CHANGE_THRESHOLD = 0.8  
STD_MULTIPLIER = 1.0        
ALERT_COOLDOWN = 900        # 15 دقیقه
HEARTBEAT_INTERVAL = 3600   # 1 ساعت

# تنظیمات اندیکاتورهای اضافی و مدیریت ریسک
ADX_THRESHOLD = 25          # حد آستانه ADX
ATR_PERIOD = 14             # دوره ATR
ATR_MULTIPLIER_SL = 1.5     # ضرایب برای استاپ لاس بر اساس ATR
TP1_MULTIPLIER = 2.0        # سطح اول TP (مثلاً ATR*2)
TP2_MULTIPLIER = 3.0        # سطح دوم TP (مثلاً ATR*3)
TP3_MULTIPLIER = 4.0        # سطح سوم TP (مثلاً ATR*4)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

last_alert_time = 0
last_heartbeat_time = 0

# =============================================================================
# تابع ارسال پیام به تلگرام
# =============================================================================
def send_telegram_message(message):
    """ارسال پیام متنی به تلگرام."""
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

# =============================================================================
# توابع پیشرفته تحلیل تکنیکال
# =============================================================================

def identify_doji_type(row, body_threshold=0.05, gravestone_threshold=0.7, dragonfly_threshold=0.7):
    """
    تشخیص انواع دوجی با تعریف سخت‌گیرانه:
    - بدنه کندل حداکثر 5٪ از رنج کندل باشد
    - برای Gravestone و Dragonfly، سایه‌ی اصلی حداقل 70٪ از رنج کندل
    """
    high = row['high']
    low = row['low']
    op = row['open']
    cl = row['close']
    candle_range = high - low
    if candle_range == 0:
        return None

    body_size = abs(cl - op)
    upper_shadow = high - max(op, cl)
    lower_shadow = min(op, cl) - low

    if body_size > body_threshold * candle_range:
        return None

    if (lower_shadow <= 0.1 * candle_range and 
        upper_shadow >= gravestone_threshold * candle_range and
        (min(op, cl) - low) <= 0.1 * candle_range):
        return "gravestone"

    if (upper_shadow <= 0.1 * candle_range and
        lower_shadow >= dragonfly_threshold * candle_range and
        (high - max(op, cl)) <= 0.1 * candle_range):
        return "dragonfly"

    if (upper_shadow >= 0.3 * candle_range and
        lower_shadow >= 0.3 * candle_range):
        return "long_legged"

    return "standard"

def identify_pin_bar(row, body_max_ratio=0.25, tail_min_ratio=0.7):
    """
    تشخیص پین‌بار صعودی یا نزولی با شرایط سخت‌تر:
    - بدنه کندل حداکثر 25٪ از رنج کندل باشد
    - سایه‌ی اصلی حداقل 70٪ از رنج کندل
    """
    high = row['high']
    low = row['low']
    op = row['open']
    cl = row['close']
    candle_range = high - low
    if candle_range == 0:
        return None

    body_size = abs(cl - op)
    upper_shadow = high - max(op, cl)
    lower_shadow = min(op, cl) - low

    if body_size > body_max_ratio * candle_range:
        return None

    if (lower_shadow >= tail_min_ratio * candle_range and
        upper_shadow <= 0.1 * candle_range and
        cl > op):
        return "bullish_pin"

    if (upper_shadow >= tail_min_ratio * candle_range and
        lower_shadow <= 0.1 * candle_range and
        cl < op):
        return "bearish_pin"

    return None

def detect_advanced_divergence(df, rsi_period=14, pivot_size=3,
                               price_diff_threshold=1.2,
                               rsi_diff_threshold=6.0,
                               rsi_zone_filter=True):
    """
    تشخیص واگرایی پیشرفته:
    - اختلاف قیمت حداقل 1.2٪ و اختلاف RSI حداقل 6 واحد
    - برای نزولی: RSI > 60؛ برای صعودی: RSI < 40
    """
    df['rsi'] = ta.rsi(df['close'], length=rsi_period)
    window_size = 20
    if len(df) < window_size:
        return None
    df_window = df.iloc[-window_size:].reset_index(drop=True)
    
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
    rsi_peaks = find_peaks(df_window['rsi'].tolist(), pivot_size, pivot_size)
    price_valleys = find_valleys(df_window['close'].tolist(), pivot_size, pivot_size)
    rsi_valleys = find_valleys(df_window['rsi'].tolist(), pivot_size, pivot_size)

    if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
        last_price_peak = price_peaks[-1]
        prev_price_peak = price_peaks[-2]
        last_rsi_peak = rsi_peaks[-1]
        prev_rsi_peak = rsi_peaks[-2]
        price_diff_percent = (df_window['close'].iloc[last_price_peak] - df_window['close'].iloc[prev_price_peak]) / df_window['close'].iloc[prev_price_peak] * 100
        rsi_diff = df_window['rsi'].iloc[last_rsi_peak] - df_window['rsi'].iloc[prev_rsi_peak]
        if price_diff_percent >= price_diff_threshold and rsi_diff <= -rsi_diff_threshold:
            if (not rsi_zone_filter) or (df_window['rsi'].iloc[last_rsi_peak] > 60):
                return "واگرایی نزولی (Bearish Divergence)"
    if len(price_valleys) >= 2 and len(rsi_valleys) >= 2:
        last_price_valley = price_valleys[-1]
        prev_price_valley = price_valleys[-2]
        last_rsi_valley = rsi_valleys[-1]
        prev_rsi_valley = rsi_valleys[-2]
        price_diff_percent = (df_window['close'].iloc[last_price_valley] - df_window['close'].iloc[prev_price_valley]) / df_window['close'].iloc[prev_price_valley] * 100
        rsi_diff = df_window['rsi'].iloc[last_rsi_valley] - df_window['rsi'].iloc[prev_rsi_valley]
        if price_diff_percent <= -price_diff_threshold and rsi_diff >= rsi_diff_threshold:
            if (not rsi_zone_filter) or (df_window['rsi'].iloc[last_rsi_valley] < 40):
                return "واگرایی صعودی (Bullish Divergence)"
    return None

def find_support_resistance(df, window=5):
    """محاسبه سطوح حمایت و مقاومت."""
    try:
        df['support'] = df['low'].rolling(window=window).min()
        df['resistance'] = df['high'].rolling(window=window).max()
        return df
    except Exception as e:
        logging.error("خطا در find_support_resistance: " + str(e))
        return df

def find_trendline(df):
    """تشخیص روند ساده بر اساس آخرین سه کندل."""
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

def is_big_green_candle(row, threshold=2.0):
    """تشخیص کندل سبز قدرتمند."""
    try:
        if row['open'] == 0:
            return False
        body_pct = (row['close'] - row['open']) / row['open'] * 100
        return body_pct >= threshold
    except Exception as e:
        logging.error(f"خطا در is_big_green_candle: {e}")
        return False

def is_price_rise_above_threshold(df, threshold=2.0):
    """آیا تغییر قیمت کندل آخر بیش از threshold% است؟"""
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
# توابع تشخیص جهش (Spike)
# =============================================================================

def calculate_volume_threshold(candles):
    volumes = [candle.get('volume', 0) for candle in candles[:-1]]
    return mean(volumes) * VOLUME_MULTIPLIER

def calculate_price_spike(candles):
    close_prices = [candle['close'] for candle in candles[:-1]]
    if len(close_prices) < 2:
        return 0, None
    price_changes = []
    for i in range(1, len(close_prices)):
        change = (close_prices[i] - close_prices[i-1]) / close_prices[i-1] * 100
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

# =============================================================================
# دریافت داده از CryptoCompare (کندل‌های 15 دقیقه‌ای)
# =============================================================================

def get_bitcoin_data():
    """دریافت داده BTC/USDT از CryptoCompare."""
    import requests
    url = "https://min-api.cryptocompare.com/data/v2/histominute"
    params = {
        'fsym': 'BTC',
        'tsym': 'USDT',
        'limit': NUM_CANDLES,
        'aggregate': 15,
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

def get_symbol_data(symbol, timeframe, limit=60):
    """دریافت داده سایر نمادها از CryptoCompare."""
    import requests
    try:
        if timeframe == '15m':
            url = "https://min-api.cryptocompare.com/data/v2/histominute"
            aggregate = 15
            limit = 60
        elif timeframe == '1h':
            url = "https://min-api.cryptocompare.com/data/v2/histominute"
            aggregate = 1
            limit = 60
        elif timeframe == '1d':
            url = "https://min-api.cryptocompare.com/data/v2/histohour"
            aggregate = 1
            limit = 24
        else:
            raise ValueError("تایم‌فریم پشتیبانی نمی‌شود. فقط '15m'، '1h' یا '1d' مجاز است.")

        if symbol.endswith("USDT"):
            fsym = symbol[:-4]
            tsym = "USDT"
        else:
            fsym, tsym = symbol.split('/')

        params = {
            'fsym': fsym,
            'tsym': tsym,
            'limit': limit,
            'aggregate': aggregate,
            'api_key': CRYPTOCOMPARE_API_KEY
        }
        response = requests.get(url, params=params, timeout=10)
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
        logging.error(f"خطا در get_price_data برای {symbol}: {e}")
        return pd.DataFrame()

# =============================================================================
# توابع نظارت و تحلیل
# =============================================================================

def monitor_bitcoin():
    global last_alert_time, last_heartbeat_time
    logging.info("شروع نظارت بر BTC/USDT (15m) از CryptoCompare...")
    send_telegram_message("سیستم نظارت BTC/USDT فعال شد (کندل‌های 15 دقیقه‌ای - منبع CryptoCompare).")
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
            
            if time.time() - last_heartbeat_time >= HEARTBEAT_INTERVAL:
                send_telegram_message("سیستم نظارت BTC/USDT همچنان فعال است (منبع CryptoCompare).")
                last_heartbeat_time = time.time()

            logging.info("چرخه نظارت BTC/USDT تکمیل شد.")
            time.sleep(900)
        except Exception as ex:
            logging.error("خطای غیرمنتظره در monitor_bitcoin: " + str(ex))
            time.sleep(60)

def analyze_symbol(symbol, timeframe='15m'):
    df = get_symbol_data(symbol, timeframe, limit=60)
    if df.empty or len(df) < 3:
        return f"تحلیل بازار برای {symbol}: داده‌های کافی دریافت نشد."
    
    # محاسبات پایه
    df = find_support_resistance(df)
    trend = find_trendline(df)
    divergence = detect_advanced_divergence(df)
    rsi_val = df['rsi'].iloc[-1] if 'rsi' in df.columns else None

    # محاسبه MACD و ADX برای تایید روند
    macd_df = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd_df['MACD_12_26_9']
    df['MACD_signal'] = macd_df['MACDs_12_26_9']
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['ADX'] = adx_df['ADX_14']
    df['DIp'] = adx_df['DMP_14']
    df['DIN'] = adx_df['DMN_14']

    # محاسبه ATR برای مدیریت ریسک
    atr_val = ta.atr(df['high'], df['low'], df['close'], length=ATR_PERIOD).iloc[-1]

    # تشخیص الگوهای کندلی
    doji_types = df.apply(identify_doji_type, axis=1)
    latest_doji = doji_types.iloc[-1]
    pin_bar = df.apply(identify_pin_bar, axis=1).iloc[-1]
    big_green = df.apply(is_big_green_candle, axis=1).iloc[-1]
    price_rise_2pct = is_price_rise_above_threshold(df, 2.0)

    # تعیین سیگنال و مدیریت ریسک
    signal = "سیگنالی یافت نشد"
    entry_price = df['close'].iloc[-1]
    sl = tp1 = tp2 = tp3 = None
    risk_message = ""

    # تعریف متغیرهای قالب‌بندی شده برای نمایش اطلاعات
    rsi_str = f"{rsi_val:.2f}" if rsi_val is not None else "N/A"
    support_str = f"{df['support'].iloc[-1]:.2f}"
    resistance_str = f"{df['resistance'].iloc[-1]:.2f}"
    macd_str = f"{df['MACD'].iloc[-1]:.2f}"
    macd_signal_str = f"{df['MACD_signal'].iloc[-1]:.2f}"
    adx_str = f"{df['ADX'].iloc[-1]:.2f}"
    entry_str = f"{entry_price:.2f}"

    # شرایط ورود پیشرفته با تایید MACD و ADX
    if pin_bar == "bullish_pin" and rsi_val is not None and rsi_val > 30:
        if (df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1] and 
            df['ADX'].iloc[-1] > ADX_THRESHOLD and 
            df['DIp'].iloc[-1] > df['DIN'].iloc[-1]):
            signal = "ورود به پوزیشن Long (Bullish Pin Bar + تایید MACD/ADX + RSI بالای 30)"
            sl = entry_price - atr_val * ATR_MULTIPLIER_SL
            tp1 = entry_price + atr_val * TP1_MULTIPLIER
            tp2 = entry_price + atr_val * TP2_MULTIPLIER
            tp3 = entry_price + atr_val * TP3_MULTIPLIER
            risk_message = (f"\nنقطه ورود: {entry_str}\n"
                            f"SL: {sl:.2f}\n"
                            f"TP1 (40%): {tp1:.2f}\n"
                            f"TP2 (30%): {tp2:.2f}\n"
                            f"TP3 (30%): {tp3:.2f}")
    elif pin_bar == "bearish_pin" and rsi_val is not None and rsi_val < 70:
        if (df['MACD'].iloc[-1] < df['MACD_signal'].iloc[-1] and 
            df['ADX'].iloc[-1] > ADX_THRESHOLD and 
            df['DIp'].iloc[-1] < df['DIN'].iloc[-1]):
            signal = "ورود به پوزیشن Short (Bearish Pin Bar + تایید MACD/ADX + RSI زیر 70)"
            sl = entry_price + atr_val * ATR_MULTIPLIER_SL
            tp1 = entry_price - atr_val * TP1_MULTIPLIER
            tp2 = entry_price - atr_val * TP2_MULTIPLIER
            tp3 = entry_price - atr_val * TP3_MULTIPLIER
            risk_message = (f"\nنقطه ورود: {entry_str}\n"
                            f"SL: {sl:.2f}\n"
                            f"TP1 (40%): {tp1:.2f}\n"
                            f"TP2 (30%): {tp2:.2f}\n"
                            f"TP3 (30%): {tp3:.2f}")
    elif latest_doji is not None:
        if latest_doji == "gravestone":
            signal = "الگوی Gravestone Doji شناسایی شد (فشار فروش)"
        elif latest_doji == "dragonfly":
            signal = "الگوی Dragonfly Doji شناسایی شد (فشار خرید)"
        else:
            signal = "الگوی دوجی شناسایی شد"
    elif divergence is not None:
        signal = f"واگرایی شناسایی شد: {divergence}"
    elif big_green:
        signal = "کندل صعودی قدرتمند شناسایی شد (Big Green Candle)"
    elif price_rise_2pct:
        signal = "افزایش قیمت بیش از ۲٪ در کندل اخیر"
    
    message = f"""
تحلیل بازار برای {symbol}:
- قیمت فعلی: {entry_str}
- حمایت: {support_str}
- مقاومت: {resistance_str}
- خط روند: {trend}
- RSI: {rsi_str}
- MACD: {macd_str} | سیگنال: {macd_signal_str}
- ADX: {adx_str}
- سیگنال: {signal}{risk_message}
"""
    return message

def multi_symbol_analysis_loop():
    symbols = [
        'BTCUSDT', 'ETHUSDT', 'SHIBUSDT', 'NEARUSDT',
        'SOLUSDT', 'DOGEUSDT', 'BNBUSDT'
    ]
    while True:
        try:
            for symbol in symbols:
                logging.info(f"در حال بررسی {symbol}...")
                try:
                    analysis_message = analyze_symbol(symbol, '15m')
                    logging.info(f"نتیجه تحلیل {symbol}: {analysis_message.strip()}")
                    if "سیگنال:" in analysis_message and "سیگنالی یافت نشد" not in analysis_message:
                        send_telegram_message(analysis_message)
                except Exception as e:
                    logging.error(f"خطا در بررسی {symbol}: {e}")
            logging.info("چرخه تحلیل چند ارز تکمیل شد.")
            time.sleep(900)
        except Exception as ex:
            logging.error("خطای غیرمنتظره در multi_symbol_analysis_loop: " + str(ex))
            time.sleep(60)

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
