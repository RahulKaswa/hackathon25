import requests
import pandas as pd
from prophet import Prophet
from flask import Flask, jsonify
import time

app = Flask(__name__)

PROMETHEUS_URL = "http://prometheus:9090/api/v1/query_range"
QUERY = "rate(http_requests_total[1m])"
STEP = "60"
RANGE_MINUTES = 60

def get_prometheus_data():
    end = int(time.time())
    start = end - RANGE_MINUTES * 60
    params = {
        "query": QUERY,
        "start": start,
        "end": end,
        "step": STEP
    }
    resp = requests.get(PROMETHEUS_URL, params=params)
    result = resp.json()["data"]["result"]
    if not result:
        return pd.DataFrame(columns=["ds", "y"])
    values = result[0]["values"]
    df = pd.DataFrame(values, columns=["ds", "y"])
    df["ds"] = pd.to_datetime(df["ds"], unit="s")
    df["y"] = df["y"].astype(float)
    return df

def predict(df):
    if df.empty:
        return {"predicted_load": 0}
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=5, freq='min')
    forecast = model.predict(future)
    yhat = forecast.iloc[-1]["yhat"]
    return {"predicted_load": max(0, yhat)}

@app.route('/predict')
def predict_endpoint():
    df = get_prometheus_data()
    result = predict(df)
    return jsonify(result)

@app.route('/metrics')
def metrics():
    # Expose predicted value as a Prometheus metric
    df = get_prometheus_data()
    result = predict(df)
    metric = f'predicted_load {result["predicted_load"]}\n'
    return metric, 200, {'Content-Type': 'text/plain; version=0.0.4; charset=utf-8'}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
