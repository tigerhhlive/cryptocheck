import os
import time
import logging
import threading
import requests
import pandas as pd
import pandas_ta as ta
from flask import Flask
from datetime import datetime

app = Flask(__name__)

# === تنظیمات محیطی ===
CRYPTOCOMPARE_API_KEY = os.environ['CRYPTOCOMPARE_API_KEY']
TELEGRAM_BOT_TOKEN    = os.environ['TELEGRAM_BOT_TOKEN']
TELEGRAM_CHAT_ID      = os.environ['TELEGRAM_CHAT_ID']

# === پارامترهای استراتژی ===
EMA_LEN       = 9
RSI_PERIOD    = 14        # می‌تونین غیرفعالش کنین
OB_LOOKBACK   = 10        # طول پویوت برای Order Block
ATR_PERIOD    = 14        # فقط برای SL/TP
SL_ATR_MULT   = 1.0
TP1_ATR_MULT  = 1.0
TP2_ATR_MULT  = 2.0

CHECK_INTERVAL     = 600
HEARTBEAT_INTERVAL = 7200
SLEEP_HOURS        = (0, 7)    # تهران: بین ۰ تا ۷ صبح

MIN_BARS = max(EMA_LEN, RSI_PERIOD, ATR_PERIOD, OB_LOOKBACK * 2 + 1)

# === وضعیت داخلی ===
last_signals     = {}   # cooldown per symbol+dir+bar
open_positions   = {}   # برای گزارش روزانه
daily_signals    = 0
daily_wins       = 0
daily_losses     = 0

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')

def send_telegram(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    resp = requests.post(url, json=payload, timeout=10)
    if resp.status_code != 200:
        logging.error(f"Telegram error: {resp.text}")

def get_data(symbol: str, timeframe: str='15m') -> pd.DataFrame:
    agg = 5 if timeframe=='5m' else 15
    params = {
        'fsym': symbol[:-4], 'tsym': 'USDT',
        'limit': 60, 'aggregate': agg,
        'api_key': CRYPTOCOMPARE_API_KEY
    }
    resp = requests.get("https://min-api.cryptocompare.com/data/v2/histominute",
                        params=params, timeout=10).json()
    data = resp['Data']['Data']
    df = pd.DataFrame(data)
    df['timestamp']    = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={'volumeto':'volume'}, inplace=True)
    return df[['timestamp','open','high','low','close','volume']]

def check_cooldown(symbol: str, direction: str, bar_index: int) -> bool:
    key = f"{symbol}_{direction}"
    if last_signals.get(key)==bar_index:
        return False
    last_signals[key] = bar_index
    return True

def analyze_symbol(symbol: str, timeframe: str='15m') -> str | None:
    """
    تشخیص Order-Block + کراس EMA9
    برمی‌گردونه پیام Markdown یا None
    """
    global daily_signals

    df = get_data(symbol, timeframe)
    if len(df) < MIN_BARS:
        return None

    # اندیکاتورها
    df['EMA9'] = ta.ema(df['close'], length=EMA_LEN)
    df['RSI']  = ta.rsi(df['close'], length=RSI_PERIOD)
    df['ATR']  = ta.atr(df['high'], df['low'], df['close'], length=ATR_PERIOD)

    # پیووت‌ها (Order Block)
    df['OB_high'] = ta.pivothigh(df['high'], left=OB_LOOKBACK, right=OB_LOOKBACK)
    df['OB_low']  = ta.pivotlow (df['low'],  left=OB_LOOKBACK, right=OB_LOOKBACK)

    # می‌گیریم آخرین پیووت معتبر قبل از کندل فعلی
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    idx  = df.index[-1]

    # جستجوی آخرین سطح OB_high/low
    obh = df['OB_high'].dropna()
    obl = df['OB_low'].dropna()
    last_pivot_high = obh.iloc[-1] if len(obh)>0 else None
    last_pivot_low  = obl.iloc[-1] if len(obl)>0 else None

    # شرط شکست و کراس EMA و فیلتر RSI
    signal = None
    direction = None

    # لانگ: بسته شدن کندل روی آخرین OB_low و بالای EMA9
    if last_pivot_low is not None:
        if prev['close'] >= last_pivot_low and curr['close'] < last_pivot_low \
           and curr['close'] < curr['EMA9'] \
           and curr['RSI'] < 50:
            direction = 'Short'
    # شورت: بسته شدن کندل زیر OB_high و زیر EMA9
    if last_pivot_high is not None:
        if prev['close'] <= last_pivot_high and curr['close'] > last_pivot_high \
           and curr['close'] > curr['EMA9'] \
           and curr['RSI'] > 50:
            direction = 'Long'

    if direction is None:
        logging.info(f"{symbol}: No OB/EMA9 signal")
        return None

    # cooldown
    if not check_cooldown(symbol, direction, idx):
        return None

    entry = curr['close']
    atr   = curr['ATR']

    # SL/TP
    if direction=='Long':
        sl  = entry - SL_ATR_MULT  * atr
        tp1 = entry + TP1_ATR_MULT * atr
        tp2 = entry + TP2_ATR_MULT * atr
        emoji = "🟢 BUY"
    else:
        sl  = entry + SL_ATR_MULT  * atr
        tp1 = entry - TP1_ATR_MULT * atr
        tp2 = entry - TP2_ATR_MULT * atr
        emoji = "🔴 SELL"

    daily_signals += 1
    open_positions[symbol] = {
        'direction':direction, 'sl':sl, 'tp1':tp1, 'tp2':tp2
    }

    stars = "🔥🔥🔥"
    msg = (
        f"🚨 *This Is AI Signal Alert*\n"
        f"*Symbol:* `{symbol}`\n"
        f"*Signal:* {emoji} *MARKET*\n"
        f"*Entry:* `{entry:.6f}`\n"
        f"*Stop Loss:* `{sl:.6f}`   *TP1:* `{tp1:.6f}`   *TP2:* `{tp2:.6f}`\n"
        f"*EMA9:* {curr['EMA9']:.4f}   *RSI:* {curr['RSI']:.1f}\n"
        f"*Strength:* {stars}"
    )
    return msg

def analyze_symbol_mtf(symbol: str) -> str | None:
    """Multi-timeframe: هم ۵ دقیقه و هم ۱۵ دقیقه باید موافق باشند"""
    m5  = analyze_symbol(symbol, '5m')
    m15 = analyze_symbol(symbol, '15m')
    if m5 and m15 and (("BUY" in m5 and "BUY" in m15) or ("SELL" in m5 and "SELL" in m15)):
        return m15
    return None

def check_and_alert(symbol: str):
    logging.info(f"🔍 Checking {symbol} …")
    msg = analyze_symbol_mtf(symbol)
    if msg:
        send_telegram(msg)
        logging.info(f"✅ Sent signal for {symbol}")

def monitor_positions():
    """کنترل بسته شدن پوزیشن‌ها برای گزارش روزانه"""
    global daily_wins, daily_losses
    while True:
        df = None
        for sym, pos in list(open_positions.items()):
            df = get_data(sym, '15m')
            last = df['close'].iloc[-1]
            dir  = pos['direction']
            if dir=='Long':
                if last >= pos['tp2']:
                    daily_wins += 1;  open_positions.pop(sym)
                elif last <= pos['sl']:
                    daily_losses += 1; open_positions.pop(sym)
            else:
                if last <= pos['tp2']:
                    daily_wins += 1;  open_positions.pop(sym)
                elif last >= pos['sl']:
                    daily_losses += 1; open_positions.pop(sym)
        time.sleep(60)

def report_daily():
    """گزارش روزانه در ۲۳:۵۵ تهران"""
    total = daily_wins + daily_losses
    wr = round(daily_wins/total*100,1) if total>0 else 0.0
    send_telegram(
        f"📊 *Daily Performance Report*\n"
        f"Total Signals: {daily_signals}\n"
        f"🎯 Wins : {daily_wins}\n"
        f"❌ Losses: {daily_losses}\n"
        f"📈 Winrate: {wr}%"
    )

def monitor():
    symbols = [
        "BTCUSDT","ETHUSDT","DOGEUSDT","BNBUSDT","XRPUSDT",
        "RENDERUSDT","TRUMPUSUSDT","FARTCOINUSDT","XLMUSDT",
        "SHIBUSDT","ADAUSDT","NOTUSDT","PROMUSDT","PENDLEUSDT"
    ]
    last_hb = 0
    while True:
        now = datetime.utcnow()
        te_hr = (now.hour+3)%24; te_mn = now.minute

        # خواب ربات
        if SLEEP_HOURS[0] <= te_hr < SLEEP_HOURS[1]:
            time.sleep(60)
            continue

        # هارت‌بیت
        if time.time() - last_hb > HEARTBEAT_INTERVAL:
            send_telegram("🤖 *Bot live and scanning.*")
            last_hb = time.time()

        # اسکن همه نمادها
        threads = []
        for s in symbols:
            t = threading.Thread(target=check_and_alert, args=(s,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        # گزارش روزانه
        if te_hr==23 and te_mn>=55:
            report_daily()

        time.sleep(CHECK_INTERVAL)

@app.route('/')
def home():
    return "✅ Crypto Signal Bot is running."

if __name__=='__main__':
    threading.Thread(target=monitor, daemon=True).start()
    threading.Thread(target=monitor_positions, daemon=True).start()
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
