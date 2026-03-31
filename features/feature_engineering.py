import numpy as np


def compute_rsi(df, period=14):

    delta = df['Close'].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi


def add_features(df):

    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()

    df['RSI'] = compute_rsi(df)

    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()

    df['Volatility'] = df['Close'].pct_change().rolling(10).std()

    df['Volume_Spike'] = df['Volume'] > df['Volume'].rolling(20).mean()

    df.dropna(inplace=True)

    return df
