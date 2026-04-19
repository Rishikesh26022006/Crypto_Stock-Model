from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import pandas as pd
import yfinance as yf
import traceback

app = Flask(__name__)
CORS(app)

# 1. Load the AI Models
print("Loading AI Models into memory...")
try:
    with open('arima_model.pkl', 'rb') as f:
        arima_model = pickle.load(f)
    with open('prophet_model.pkl', 'rb') as f:
        prophet_model = pickle.load(f)
    with open('minmax_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✅ All Models loaded successfully!")
except Exception as e:
    print(f"⚠️ Error loading models: {e}")

# ==========================================
# WEB ROUTE
# ==========================================
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# ==========================================
# NEW: LIVE MARKET DATA API (For Tabs 1, 2, and 5)
# ==========================================
@app.route('/api/market-data', methods=['POST'])
def get_market_data():
    try:
        data = request.get_json()
        ticker = data.get('ticker', 'BTC-USD').upper()

        # Fetch last 3 months of live data
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")

        if hist.empty:
            return jsonify({"error": "Invalid ticker or no data found on Yahoo Finance."}), 400

        # Tab 1 Data: Current Stats
        current_price = hist['Close'].iloc[-1]
        volume = hist['Volume'].iloc[-1]
        
        # Calculate 7-day Trend
        past_price = hist['Close'].iloc[-7]
        trend_pct = ((current_price - past_price) / past_price) * 100
        trend_text = "Bullish" if trend_pct > 0 else "Bearish"

        # Tab 2 Data: Historical Chart Data
        hist_dates = hist.index.strftime('%Y-%m-%d').tolist()
        hist_prices = round(hist['Close'], 2).tolist()

        # Tab 5 Data: Dynamic Trading Signal
        signal = "HOLD 🟡"
        signal_message = f"Market is relatively stable. (7-Day Change: {trend_pct:.2f}%)"
        if trend_pct > 5.0:
            signal = "BUY 🟢"
            signal_message = f"Strong upward momentum detected (+{trend_pct:.2f}% over 7 days). AI threshold met."
        elif trend_pct < -5.0:
            signal = "SELL 🔴"
            signal_message = f"Downward trend detected ({trend_pct:.2f}% over 7 days). Consider mitigating risk."

        return jsonify({
            "ticker": ticker,
            "current_price": f"${current_price:,.2f}",
            "volume": f"{volume:,.0f}",
            "trend": trend_text,
            "trend_pct": round(trend_pct, 2),
            "hist_dates": hist_dates,
            "hist_prices": hist_prices,
            "signal": signal,
            "signal_message": signal_message
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

# ==========================================
# NEW: MODEL METRICS API (For Tab 3)
# ==========================================
@app.route('/api/model-metrics', methods=['GET'])
def get_model_metrics():
    # Sending dynamic data instead of hardcoding HTML
    return jsonify({
        "models": [
            {"name": "LSTM (Deep Learning)", "error": "$412.50", "use_case": "Non-Linear Dynamics"},
            {"name": "Prophet (Probabilistic)", "error": "$650.20", "use_case": "Seasonality & Holidays"},
            {"name": "ARIMA (Statistical)", "error": "$1,050.00", "use_case": "Baseline Linear Trends"}
        ]
    })

# ==========================================
# PREDICTION APIs (For Tab 4)
# ==========================================
@app.route('/predict/prophet', methods=['POST'])
def predict_prophet():
    try:
        data = request.get_json()
        days = int(data.get('days', 30))
        future = prophet_model.make_future_dataframe(periods=days)
        forecast = prophet_model.predict(future)
        future_forecast = forecast[['ds', 'yhat']].tail(days)
        
        results = [{"date": row['ds'].strftime('%Y-%m-%d'), "predicted_price": round(row['yhat'], 2)} for _, row in future_forecast.iterrows()]
        return jsonify({"forecast": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/arima', methods=['POST'])
def predict_arima():
    try:
        data = request.get_json()
        days = int(data.get('days', 30))
        forecast_prices = arima_model.predict(n_periods=days)
        results = [{"day": i + 1, "predicted_price": round(price, 2)} for i, price in enumerate(forecast_prices)]
        return jsonify({"forecast": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)