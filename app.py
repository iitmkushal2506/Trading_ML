from flask import Flask, render_template
from datetime import datetime, time
import pytz

from data.data_pipeline import prepare_data
from models.model import train_model
from signals.signal_generator import generate_signal

app = Flask(__name__)

# Indian timezone
ist = pytz.timezone("Asia/Kolkata")

calls_history = []
last_date = None


def market_open():
    now = datetime.now(ist).time()

    start = time(9, 20)
    end = time(15, 30)

    return start <= now <= end


def create_target(df):
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df


def generate_calls():

    data = prepare_data()

    table = []

    for stock, df in data.items():

        if df.empty:
            continue

        df = create_target(df)

        if len(df) < 50:
            continue

        X = df.drop('Target', axis=1)
        y = df['Target']

        model = train_model(X, y)

        signal, confidence = generate_signal(model, df)

        entry_series = df['Close'].iloc[-1]

        try:
            entry = float(entry_series.iloc[0])
        except:
            entry = float(entry_series)

        stop = entry * 0.995
        target = entry * 1.01

        table.append({
            "stock": stock,
            "signal": signal,
            "entry": round(entry, 2),
            "stop": round(stop, 2),
            "target": round(target, 2),
            "confidence": round(confidence, 2)
        })

    return table


@app.route("/")
def home():

    global calls_history
    global last_date

    today = datetime.now(ist).date()

    # Reset next day
    if last_date != today:
        calls_history = []
        last_date = today

    if market_open():

        data = generate_calls()

        timestamp = datetime.now(ist).strftime("%d-%m-%Y %I:%M:%S %p")

        calls_history.insert(0, {
            "time": timestamp,
            "data": data
        })

        calls_history = calls_history[:20]

        status = "MARKET OPEN"

    else:
        status = "MARKET CLOSED"

    return render_template(
        "index_live.html",
        calls=calls_history,
        status=status,
        today=today
    )


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
