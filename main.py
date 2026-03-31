from data.data_pipeline import prepare_data
from models.model import train_model
from signals.signal_generator import generate_signal
from datetime import datetime
from tabulate import tabulate


def create_target(df):

    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    return df


def run():

    print("\n============================")
    print("LIVE INTRADAY TRADE CALLS")
    print("Time:", datetime.now())
    print("============================")

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

        # Fix Future Warning
        entry_series = df['Close'].iloc[-1]

        try:
            entry = float(entry_series.iloc[0])
        except:
            entry = float(entry_series)

        stop = entry * 0.995
        target = entry * 1.01

        table.append([
            stock,
            signal,
            round(entry, 2),
            round(stop, 2),
            round(target, 2),
            round(confidence, 2)
        ])

    # Print Table
    print("\n")

    print(tabulate(
        table,
        headers=[
            "Stock",
            "Signal",
            "Entry",
            "Stop Loss",
            "Target",
            "Confidence"
        ],
        tablefmt="fancy_grid"
    ))

    return table


if __name__ == "__main__":
    run()