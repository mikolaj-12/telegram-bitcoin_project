import os
import logging
from datetime import timedelta
import numpy as np
import yfinance as yf
from joblib import load
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.io as pio
from telegram import Update
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from flask import Flask, request

# Inicjalizacja Flask dla webhooka
app = Flask(__name__)

# Logging dla debugowania
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Pobieranie tokena z zmiennych środowiskowych
BOT_TOKEN = os.getenv('TELEGRAM_TOKEN')
if not BOT_TOKEN:
    raise ValueError("TELEGRAM_TOKEN not set in environment variables")

# Cache modeli, aby ładować leniwie i oszczędzać RAM
MODEL_CACHE = {}

# ---------- Funkcje pomocnicze ----------

def get_data(typename):
    match typename:
        case '1h':
            period = '5d'
            window = 20
        case '4h':
            period = '60d'
            window = 20
        case '1d':
            period = '2y'
            window = 10

    btc = yf.Ticker('BTC-USD')
    data = btc.history(period=period, interval=typename)

    scaler = load('models/scaler.pkl')
    v_scaler = load('models/volume_scaler.pkl')

    o, c, h, l, v = data['Open'], data['Close'], data['High'], data['Low'], data['Volume']
    date = data.index[-1]

    # Skalerowanie danych
    open_data = scaler.transform(o.values.reshape(-1, 1))
    close_data = scaler.transform(c.values.reshape(-1, 1))
    high_data = scaler.transform(h.values.reshape(-1, 1))
    low_data = scaler.transform(l.values.reshape(-1, 1))
    volume_data = v_scaler.transform(v.values.reshape(-1, 1))

    x, y = [], []
    if typename != '1h':
        for i in range(len(open_data) - window):
            x.append([[open_data[j], close_data[j], high_data[j], low_data[j], volume_data[j]]
                      for j in range(i, i + window)])
            y.append([open_data[i + window], close_data[i + window], high_data[i + window],
                      low_data[i + window], volume_data[i + window]])
    else:
        for i in range(len(open_data) - window):
            x.append([[open_data[j], close_data[j], high_data[j], low_data[j]]
                      for j in range(i, i + window)])
            y.append([open_data[i + window], close_data[i + window], high_data[i + window],
                      low_data[i + window]])

    return np.array(x), np.array(y), window, scaler, v_scaler, date

def predict_price(t, n):
    x, y, window, scaler, v_scaler, date = get_data(t)

    # Leniwe ładowanie modelu
    model_key = f"{t}_model"
    if model_key not in MODEL_CACHE:
        match t:
            case '1h':
                MODEL_CACHE[model_key] = load_model('models/1h_model.keras')
                dim = 4
            case '4h':
                MODEL_CACHE[model_key] = load_model('models/4h_model.keras')
                dim = 5
            case '1d':
                MODEL_CACHE[model_key] = load_model('models/1d_model.keras')
                dim = 5
    net = MODEL_CACHE[model_key]

    # Przygotowanie ostatniego okna
    z_test = x[-1].reshape(1, window, dim)
    all_pred = []
    date_x = []
    last_date = date + timedelta(hours=2)

    for i in range(n):
        pred = net.predict(z_test, verbose=0)
        all_pred.append(pred[0])

        # Generowanie kolejnej daty
        match t:
            case '1h':
                last_date += timedelta(hours=1)
            case '4h':
                last_date += timedelta(hours=4)
            case '1d':
                last_date += timedelta(days=1)
        date_x.append(last_date.strftime(r"%d:%m %H:%M"))

        # Aktualizacja z_test
        z_test = z_test.reshape(window, dim)
        z_test = np.vstack([z_test[1:], pred])
        z_test = z_test.reshape(1, window, dim)

    all_pred = np.array(all_pred)

    o_predict = scaler.inverse_transform(all_pred[:, 0].reshape(-1, 1))
    c_predict = scaler.inverse_transform(all_pred[:, 1].reshape(-1, 1))
    h_predict = scaler.inverse_transform(all_pred[:, 2].reshape(-1, 1))
    l_predict = scaler.inverse_transform(all_pred[:, 3].reshape(-1, 1))
    if t != '1h':
        v_predict = v_scaler.inverse_transform(all_pred[:, 4].reshape(-1, 1))
    else:
        v_predict = 0

    for i in range(len(o_predict)):
        h_predict[i][0] = max(o_predict[i][0], c_predict[i][0], h_predict[i][0])
        l_predict[i][0] = min(o_predict[i][0], c_predict[i][0], l_predict[i][0])

    return o_predict, c_predict, h_predict, l_predict, v_predict, date_x

def create_candlestick_chart(date, open_, high, low, close):
    open_ = np.array(open_).flatten()
    high = np.array(high).flatten()
    low = np.array(low).flatten()
    close = np.array(close).flatten()

    fig = go.Figure(data=[go.Candlestick(
        x=date,
        open=open_,
        high=high,
        low=low,
        close=close,
        increasing_line_color='green',
        decreasing_line_color='red'
    )])

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        title='BTC Price Prediction',
        width=800,
        height=600
    )

    img_bytes = pio.to_image(fig, format='png', engine='kaleido')
    buf = io.BytesIO(img_bytes)
    buf.seek(0)
    return buf

# ---------- Handlery Telegram ----------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("1h", callback_data='timeframe_1h')],
        [InlineKeyboardButton("4h", callback_data='timeframe_4h')],
        [InlineKeyboardButton("1d", callback_data='timeframe_1d')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text('Select timeframe for BTC prediction:', reply_markup=reply_markup)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = """
🤖 BTC Prediction Bot

Commands:
/start - Begin prediction
/help - Show this help

Steps:
1. Choose timeframe (1h/4h/1d)
2. Choose number of periods (1-5)
3. Get prediction text + chart
"""
    await update.message.reply_text(text)

async def handle_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    timeframe = query.data.split('_')[1]
    context.user_data['timeframe'] = timeframe

    keyboard = [[InlineKeyboardButton(f"{i}", callback_data=f'periods_{i}')] for i in range(1, 6)]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(f'Selected: {timeframe}\nChoose number of periods:', reply_markup=reply_markup)

async def handle_periods(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    periods = int(query.data.split('_')[1])
    timeframe = context.user_data.get('timeframe')

    await query.edit_message_text("🔄 Generating prediction...")

    try:
        open_pred, close_pred, high_pred, low_pred, volume_pred, dates = predict_price(timeframe, periods)

        # Tworzenie tekstu
        text = f"📈 BTC {timeframe} Prediction ({periods} periods):\n\n"
        for i in range(periods):
            text += f"📅 {dates[i]}\n"
            text += f"🟢 Open: ${open_pred[i][0]:,.2f}\n"
            text += f"🔴 Close: ${close_pred[i][0]:,.2f}\n"
            text += f"🔵 High: ${high_pred[i][0]:,.2f}\n"
            text += f"🔴 Low: ${low_pred[i][0]:,.2f}\n"
            if timeframe != '1h':
                text += f"📊 Volume: {volume_pred[i][0]:,.0f}\n"
            text += "\n"

        await context.bot.send_message(chat_id=query.message.chat_id, text=text)

        # Wysyłanie wykresu
        chart_buffer = create_candlestick_chart(dates, open_pred, high_pred, low_pred, close_pred)
        await context.bot.send_photo(
            chat_id=query.message.chat_id,
            photo=chart_buffer,
            caption=f"📊 BTC {timeframe} Candlestick Chart"
        )

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        await context.bot.send_message(chat_id=query.message.chat_id, text=f"❌ Error: {str(e)[:200]}")

# ---------- Webhook dla Render ----------

@app.route('/webhook', methods=['POST'])
def webhook():
    update = Update.de_json(request.get_json(), application.bot)
    application.process_update(update)
    return 'OK'

@app.route('/health', methods=['GET'])
def health():
    return 'Bot is running!'

# ---------- Uruchomienie bota ----------

application = Application.builder().token(BOT_TOKEN).build()

def main():
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CallbackQueryHandler(handle_timeframe, pattern=r'^timeframe_'))
    application.add_handler(CallbackQueryHandler(handle_periods, pattern=r'^periods_'))
    logger.info("Bot starting...")

    # Ustawienie webhooka
    webhook_url = f"{os.getenv('RENDER_EXTERNAL_URL')}/webhook"
    logger.info(f"Setting webhook: {webhook_url}")
    application.bot.set_webhook(url=webhook_url)
    app.run(host='0.0.0.0', port=10000)  # Port 10000 dla Render

if __name__ == '__main__':
    main()
