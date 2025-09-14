from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('multiple_linear_model.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        location = request.form['Location']
        RnDSpend = float(request.form['RndSpend'])
        AdminSpend = float(request.form['AdminSpend'])
        MarketSpend = float(request.form['MarketSpend'])

        # One-hot encode location
        if location == 'California':
            California, Florida, NewYork = 1, 0, 0
        elif location == 'Florida':
            California, Florida, NewYork = 0, 1, 0
        else:  # New York
            California, Florida, NewYork = 0, 0, 1

        # Match input order to the model
        pred_args = [California, Florida, NewYork, RnDSpend, AdminSpend, MarketSpend]
        pred_args_arr = np.array(pred_args).reshape(1, -1)

        # Predict
        model_prediction = model.predict(pred_args_arr)
        model_prediction = round(float(model_prediction), 2)

        return render_template('predict.html', prediction=model_prediction)

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(host='0.0.0.0')
