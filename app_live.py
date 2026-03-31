from flask import Flask, render_template
from datetime import datetime, time
import pandas as pd

from data.data_pipeline import prepare_data
from models.model import train_model
from signals.signal_generator import generate_signal

app = Flask(__name__)


def market_open():
    now = datetime.now().time()

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

    if market_open():
        data = generate_calls()
        status = "MARKET OPEN"
    else:
        data = []
        status = "MARKET CLOSED"

    current_time = datetime.now()

    return render_template(
        "index_live.html",
        data=data,
        status=status,
        time=current_time
    )


# Render / Gunicorn compatible
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
