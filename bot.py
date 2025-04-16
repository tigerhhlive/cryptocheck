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

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            logging.error(f"Telegram error: {response.text}")
    except Exception as e:
        logging.error(f"Telegram exception: {e}")

def get_data(timeframe, symbol):
    url = "https://min-api.cryptocompare.com/data/v2/histominute"
    aggregate = 5 if timeframe == '5m' else 15 if timeframe == '15m' else 60  # 60 for 1 hour
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
        data = json_data.get("Data", {}).get("Data", [])
        if not data or not isinstance(data, list):
            logging.warning(f"⚠️ No valid data received for {symbol} in {timeframe}. Raw: {json_data}")
            return None
        df = pd.DataFrame(data)
        
        # اصلاحات برای داده‌های NaN و تبدیل صحیح نام‌ها
        if df.isnull().values.any():
            logging.warning("⚠️ Data contains NaN values, cleaning...")
            df = df.dropna()
        
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df['volume'] = df['volumeto']  # اصلاح نام 'volumeto' به 'volume'
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        logging.error(f"❌ Error fetching data for {symbol}: {e}")
        return None

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
    if len(df) < 3:
        return None
    prev = df.iloc[-3]
    curr = df.iloc[-2]
    if prev['close'] < prev['open'] and curr['close'] > curr['open'] and curr['close'] > prev['open'] and curr['open'] < prev['close']:
        return 'bullish_engulfing'
    if prev['close'] > prev['open'] and curr['close'] < curr['open'] and curr['open'] > prev['close'] and curr['close'] < prev['open']:
        return 'bearish_engulfing'
    return None

def check_cooldown(symbol, direction):
    key = f"{symbol}_{direction}"
    last_time = last_signals.get(key)
    now = time.time()
    if last_time and (now - last_time < SIGNAL_COOLDOWN):
        return False
    last_signals[key] = now
    return True

def analyze_symbol(symbol, timeframe='15m', fast_check=False):
    global daily_signal_count

    if fast_check:
        df = get_data(timeframe, symbol).tail(15)
    else:
        df = get_data(timeframe, symbol)

    if df is None or len(df) < 15:
        logging.info(f"{symbol}: Not enough data")
        return None, "Data too short"

    # محاسبه اندیکاتورها
    df['EMA20'] = ta.ema(df['close'], length=20)
    df['EMA50'] = ta.ema(df['close'], length=50)
    df['rsi'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'])
    if macd is None or not isinstance(macd, pd.DataFrame) or macd.isnull().all().all():
        logging.info(f"{symbol}: MACD calculation failed.")
        return None, "MACD calculation failed"

    df['MACD'] = macd['MACD_12_26_9']
    df['MACDs'] = macd['MACDs_12_26_9']

     adx = ta.adx(df['high'], df['low'], df['close'])
    if adx is None or not isinstance(adx, pd.DataFrame):
        return None, "ADX calculation failed"
    if 'ADX_14' not in adx.columns or pd.isna(adx['ADX_14'].iloc[-1]):
        logging.info(f"{symbol}: ADX calculation failed.")
        return None, "ADX calculation failed"
    df['ADX'] = adx['ADX_14']


    atr_series = ta.atr(df['high'], df['low'], df['close'])
    if atr_series is None or atr_series.isnull().all():
        logging.info(f"{symbol}: ATR calculation failed.")
        return None, "ATR calculation failed"
    df['ATR'] = atr_series

    # اضافه کردن لاگ جهت نمایش مقادیر اندیکاتورها از ردیف قبل از آخر
    rsi_val = df['rsi'].iloc[-2]
    macd_val = df['MACD'].iloc[-2]
    macds_val = df['MACDs'].iloc[-2]
    adx_val = df['ADX'].iloc[-2]
    atr_val = df['ATR'].iloc[-2]
    entry = df['close'].iloc[-2]
    logging.info(f"{symbol}: Entry={entry}, RSI={rsi_val}, MACD={macd_val}, MACDs={macds_val}, ADX={adx_val}, ATR={atr_val}")

    # تحلیل شمع‌ها
    candle = df.iloc[-2]
    confirm_candle = df.iloc[-1]
    signal_type = detect_strong_candle(candle) or detect_engulfing(df)
    pattern = signal_type.replace("_", " ").title() if signal_type else "None"
    logging.info(f"{symbol}: Detected signal type: {signal_type}, Pattern: {pattern}")

    # اصلاح ATR در صورت نیاز
    atr = max(atr_val, entry * MIN_PERCENT_RISK, MIN_ATR)

    above_ema = candle['close'] > candle['EMA20'] and candle['EMA20'] > candle['EMA50']
    below_ema = candle['close'] < candle['EMA20'] and candle['EMA20'] < candle['EMA50']

    confirmations = []
    if (signal_type and 'bullish' in signal_type and rsi_val >= 50) or (signal_type and 'bearish' in signal_type and rsi_val <= 50):
        confirmations.append("RSI")
    if ((macd_val > macds_val) if ('bullish' in str(signal_type)) else (macd_val < macds_val)):
        confirmations.append("MACD")
    if adx_val > ADX_THRESHOLD:
        confirmations.append("ADX")
    if ('bullish' in str(signal_type) and above_ema) or ('bearish' in str(signal_type) and below_ema):
        confirmations.append("EMA")

    logging.info(f"{symbol}: Confirmations: {confirmations} (Confidence: {len(confirmations)})")

    confidence = len(confirmations)
    direction = 'Long' if 'bullish' in str(signal_type) and confidence >= 3 else 'Short' if 'bearish' in str(signal_type) and confidence >= 3 else None

    if direction == 'Long' and confirm_candle['close'] <= confirm_candle['open']:
        logging.info(f"{symbol}: Confirmation candle failed for Long signal.")
        return None, "Confirmation candle failed"
    if direction == 'Short' and confirm_candle['close'] >= confirm_candle['open']:
        logging.info(f"{symbol}: Confirmation candle failed for Short signal.")
        return None, "Confirmation candle failed"

    support_zone = df['low'].rolling(window=10).min().iloc[-1]
    resistance_zone = df['high'].rolling(window=10).max().iloc[-1]
    is_near_support = entry <= support_zone * 1.02
    is_near_resistance = entry >= resistance_zone * 0.98

    if direction == 'Long' and is_near_resistance:
        logging.info(f"{symbol}: Price too close to resistance for Long signal.")
        return None, "Too close to resistance"
    if direction == 'Short' and is_near_support:
        logging.info(f"{symbol}: Price too close to support for Short signal.")
        return None, "Too close to support"

    prev_high = df['high'].iloc[-5:-2].max()
    prev_low = df['low'].iloc[-5:-2].min()
    bos_long = direction == 'Long' and candle['high'] > prev_high
    bos_short = direction == 'Short' and candle['low'] < prev_low

    if direction == 'Long' and not bos_long:
        logging.info(f"{symbol}: No bullish structure break.")
        return None, "No bullish structure break"
    if direction == 'Short' and not bos_short:
        logging.info(f"{symbol}: No bearish structure break.")
        return None, "No bearish structure break"

    if direction is None and is_near_support and candle['close'] > candle['open']:
        logging.info(f"{symbol}: Only candle condition met.")
        return None, "Candle Only"
    if direction is None and is_near_resistance and candle['close'] < candle['open']:
        logging.info(f"{symbol}: Only candle condition met.")
        return None, "Candle Only"

    if direction and not check_cooldown(symbol, direction):
        logging.info(f"{symbol} - DUPLICATE SIGNAL - Skipped due to cooldown")
        return None, "Duplicate"

    if direction:
        daily_signal_count += 1

        resistance_calc = df['high'].rolling(window=10).max().iloc[-2]
        support_calc = df['low'].rolling(window=10).min().iloc[-2]
        sl = tp1 = tp2 = None

        if direction == 'Long':
            sl = entry - atr * ATR_MULTIPLIER_SL
            tp1 = min(entry + atr * TP1_MULTIPLIER, resistance_calc)
            tp2 = tp1 + (tp1 - entry) * 1.2
        elif direction == 'Short':
            sl = entry + atr * ATR_MULTIPLIER_SL
            tp1 = max(entry - atr * TP1_MULTIPLIER, support_calc)
            tp2 = tp1 - (entry - tp1) * 1.2
        else:
            logging.info(f"{symbol}: Invalid direction encountered.")
            return None, "Invalid direction"

        rr_ratio = abs(tp1 - entry) / abs(entry - sl)
        confidence_stars = "🔥" * confidence

        message = f"""🚨 *AI Signal Alert*
*Symbol:* `{symbol}`
*Signal:* {'🟢 BUY MARKET' if direction == 'Long' else '🔴 SELL MARKET'}
*Pattern:* {pattern}
*Confirmed by:* {', '.join(confirmations) if confirmations else 'None'}
*Entry:* `{entry:.6f}`
*Stop Loss:* `{sl:.6f}`
*Target 1:* `{tp1:.6f}`
*Target 2:* `{tp2:.6f}`
*Leverage (est.):* `{rr_ratio:.2f}X`
*Signal Strength:* {confidence_stars}"""
        logging.info(f"{symbol}: Signal ready to be sent. Message: {message}")
        return {
            "symbol": symbol,
            "direction": direction,
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "confidence": confidence,
            "pattern": pattern,
            "confirmations": confirmations,
            "message": message
        }, None

    return None, None


def analyze_and_alert(sym):
    try:
        logging.info(f"🟡 Starting analysis for {sym}")
        msg, _ = analyze_symbol(sym)
        if msg:
            send_telegram_message(msg)
            global daily_hit_count
            daily_hit_count += 1
        logging.info(f"✅ Done analyzing {sym}")
    except Exception as e:
        logging.error(f"❌ Error analyzing {sym}: {e}")

def monitor():
    global daily_signal_count, daily_hit_count, last_report_day

    symbols = [
        "BTCUSDT", "ETHUSDT", "DOGEUSDT", "BNBUSDT", "XRPUSDT",
        "RENDERUSDT", "TRUMPUSDT", "FARTCOINUSDT", "XLMUSDT",
        "SHIBUSDT", "ADAUSDT", "NOTUSDT", "PROMUSDT", "PENDLEUSDT"
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

        threads = []
        for sym in symbols:
            t = threading.Thread(target=analyze_and_alert, args=(sym,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        time.sleep(CHECK_INTERVAL)

def monitor_positions():
    global tp1_count, tp2_count, sl_count, last_report_day, daily_signal_count, daily_hit_count
    while True:
        for symbol, pos in list(open_positions.items()):
            try:
                df = get_data('15m', symbol)
                current_price = df['close'].iloc[-1]
                direction = pos['direction']

                if direction == 'Long':
                    if current_price >= pos['tp2']:
                        send_telegram_message(f"✅ *{symbol} TP2 Hit* - Full Target Reached. Position Closed.")
                        tp2_count += 1
                        del open_positions[symbol]
                    elif current_price >= pos['tp1']:
                        send_telegram_message(f"🎯 *{symbol} TP1 Hit* - Consider Partial Close.")
                        tp1_count += 1
                    elif current_price <= pos['sl']:
                        send_telegram_message(f"❌ *{symbol} SL Hit* - Position Closed.")
                        sl_count += 1
                        del open_positions[symbol]

                if direction == 'Short':
                    if current_price <= pos['tp2']:
                        send_telegram_message(f"✅ *{symbol} TP2 Hit* - Full Target Reached. Position Closed.")
                        tp2_count += 1
                        del open_positions[symbol]
                    elif current_price <= pos['tp1']:
                        send_telegram_message(f"🎯 *{symbol} TP1 Hit* - Consider Partial Close.")
                        tp1_count += 1
                    elif current_price >= pos['sl']:
                        send_telegram_message(f"❌ *{symbol} SL Hit* - Position Closed.")
                        sl_count += 1
                        del open_positions[symbol]
            except Exception as e:
                logging.error(f"Monitor error for {symbol}: {e}")

        now = datetime.utcnow()
        tehran_hour = (now.hour + 3) % 24
        tehran_min = now.minute
        current_day = now.date()

        if tehran_hour == 23 and tehran_min >= 55 and current_day != last_report_day:
            total = daily_signal_count
            winrate = round(((tp1_count + tp2_count) / total) * 100, 1) if total > 0 else 0.0
            report = f"""📊 *Daily Performance Report*
Total Signals: {total}
🎯 TP1 Hit: {tp1_count}
✅ TP2 Hit: {tp2_count}
❌ SL Hit: {sl_count}
📈 Estimated Winrate: {winrate}%"""
            send_telegram_message(report)
            last_report_day = current_day
            daily_signal_count = 0
            daily_hit_count = 0
            tp1_count = 0
            tp2_count = 0
            sl_count = 0
            send_telegram_message("😴 Bot going to sleep. See you tomorrow!")

        time.sleep(MONITOR_INTERVAL)

# شروع و اجرای برنامه
if __name__ == '__main__':
    threading.Thread(target=monitor, daemon=True).start()
    threading.Thread(target=monitor_positions, daemon=True).start()
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
