import os
import logging
import requests
import pandas as pd
from datetime import datetime

# تنظیمات لاگینگ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CRYPTOCOMPARE_API_KEY = os.environ.get('CRYPTOCOMPARE_API_KEY')

def send_telegram_message(message):
    # این تابع برای ارسال پیام به تلگرام است
    pass

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

    try:
        res = requests.get(url, params=params, timeout=10)
        json_data = res.json()

        # لاگ کردن پاسخ کامل API برای بررسی
        logging.info(f"API response for {symbol}: {json_data}")

        # اگر داده‌ها صحیح بودند، نمونه‌ای از داده‌ها را چاپ می‌کنیم
        if "Data" in json_data and "Data" in json_data["Data"]:
            logging.info(f"Valid data received for {symbol}: {json_data['Data']['Data'][:5]}")  # نمایش ۵ ردیف اول
        else:
            logging.warning(f"⚠️ Invalid or no data received for {symbol}.")
            send_telegram_message(f"⚠️ No valid data for {symbol}. Please check the API response.")
            return None

        # پردازش داده‌ها
        df = pd.DataFrame(json_data["Data"]["Data"])

        # چک کردن برای مقادیر نال یا داده‌های ناقص
        if df.empty or df.isnull().any().any():
            logging.warning(f"⚠️ DataFrame is empty or contains null values for {symbol}")
            send_telegram_message(f"⚠️ Null values detected in data for {symbol}")
            return None

        # اصلاح نام ستون‌ها و محاسبه حجم
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df['volume'] = df['volumefrom'] + df['volumeto']  # محاسبه صحیح حجم

        # بررسی برای هر ستون مورد نیاز
        if not all(col in df.columns for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume']):
            logging.warning(f"⚠️ Missing required columns for {symbol}.")
            return None

        # بازگشت DataFrame پردازش‌شده با ستون‌های مورد نیاز
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    except Exception as e:
        logging.error(f"❌ Error fetching data for {symbol}: {e}")
        return None

# تست برای یک نماد خاص (مثلاً BTCUSDT)
symbols_to_test = ['BTCUSDT', 'SHIBUSDT', 'PENDLEUSDT', 'XLMUSDT']
for symbol in symbols_to_test:
    get_data('15m', symbol)
