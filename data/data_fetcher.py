import yfinance as yf
from config import STOCKS, INTERVAL, PERIOD


def fetch_all():

    data = {}

    for stock in STOCKS:

        df = yf.download(
            f"{stock}.NS",
            interval=INTERVAL,
            period=PERIOD
        )

        df.dropna(inplace=True)

        data[stock] = df

    return data
