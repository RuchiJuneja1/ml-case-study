from flask import Flask
from flask import request, make_response
from sklearn.externals import joblib
import pandas as pd


import predict_main
import os

app = Flask(__name__)


# load model file and scalar file
current_dir = os.path.dirname(__file__)
model_file = os.path.join(current_dir, "../data/prediction_model.sav")
loaded_model = joblib.load(model_file)
model_scaler_file = os.path.join(current_dir, "../data/scaler.sav")
model_scaler = joblib.load(model_scaler_file)


# api entry point: generates a HTML form to upload dataset file
@app.route('/')
def form():
    return """
        <html>
            <body>
                <h1>Upload dataset file to get predictions</h1>

                <form action="/predict" method="post" enctype="multipart/form-data">
                    <input type="file" name="data_file" />
                    <input type="submit" />
                </form>
            </body>
        </html>
    """



def send_response(dataframe):
	'''

	:param dataframe: resulting dataframe to be returned via CSV
	:return: csv with <uuid>;<pd> or error csv with error message
	'''
	resp = make_response(dataframe.to_csv())
	resp.headers["Content-Disposition"] = "attachment; filename=export.csv"
	resp.headers["Content-Type"] = "text/csv"
	return resp


@app.route('/predict', methods=["POST"])
def predict_api():
	'''

	:return: csv file for the records where default is null. CSV format: <uuid>;<pd> or error csv with error message
	'''
	try:
		dataset = pd.read_csv(request.files.get('data_file'), sep=';')
	except pd.io.common.EmptyDataError:
		error_df = pd.DataFrame(columns=['error_message'])
		error_df.loc[0] = "Blank file uploaded"
		return send_response(error_df)
	default_prediction_df = predict_main.predict_default(loaded_model, dataset, model_scaler)
	return send_response(default_prediction_df)



if __name__ == '__main__':
	app.run()
