from flask import Flask, render_template
from flask import request, jsonify, make_response
from sklearn.externals import joblib
import pandas as pd


import predict_main
import os

import io
import csv

app = Flask(__name__)


current_dir = os.path.dirname(__file__)
# dataset_file = os.path.join(current_dir, "../data/dataset.csv")
# data_file = pd.read_csv(dataset_file, sep=";")
model_file = os.path.join(current_dir, "../data/prediction_model.sav")
loaded_model = joblib.load(model_file)
model_scaler_file = os.path.join(current_dir, "../data/scaler.sav")
model_scaler = joblib.load(model_scaler_file)


@app.route('/')
def form():
    return """
        <html>
            <body>
                <h1>Transform a file demo</h1>

                <form action="/predict" method="post" enctype="multipart/form-data">
                    <input type="file" name="data_file" />
                    <input type="submit" />
                </form>
            </body>
        </html>
    """



@app.route('/predict', methods=["POST"])
def predict_api():
	dataset = pd.read_csv(request.files.get('data_file'), sep=';')
	default_prediction_df = predict_main.predict_default(loaded_model, dataset, model_scaler)
	resp = make_response(default_prediction_df.to_csv())
	resp.headers["Content-Disposition"] = "attachment; filename=export.csv"
	resp.headers["Content-Type"] = "text/csv"
	return resp


# @app.route('/', methods=['GET'])
# def predict_api():
# 	obj = predict_main.predict_default(loaded_model, data_file, model_scaler)
# 	return jsonify(obj)


if __name__ == '__main__':
	app.run()
