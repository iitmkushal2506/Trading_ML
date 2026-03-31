def generate_signal(model, df):

    latest = df.iloc[-1]

    X = latest.drop('Target')

    pred = model.predict([X])[0]

    prob = model.predict_proba([X])[0]

    confidence = max(prob)

    if pred == 1:
        signal = "BUY"
    else:
        signal = "SELL"

    return signal, confidence
