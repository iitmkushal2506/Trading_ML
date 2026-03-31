from data.data_fetcher import fetch_all
from features.feature_engineering import add_features


def prepare_data():

    data = fetch_all()

    processed = {}

    for stock, df in data.items():

        processed[stock] = add_features(df)

    return processed
