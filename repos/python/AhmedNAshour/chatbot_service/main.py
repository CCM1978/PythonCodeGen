# from chefboost import Chefboost as chef
# import pandas as pd
# import pickle
# import joblib


# df = pd.read_csv("Cleaned-Data.csv")
# print(df.head())
# config = {'algorithm': 'ID3'}
# model = chef.fit(df, config)
# chef.save_model(model, "model.pkl")

# cols = ['Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat',
#         'Pains', 'Nasal-Congestion', 'Runny-Nose', 'Diarrhea', 'Smell and taste loss']
# joblib.dump(cols, 'model_cols.pkl')

import traceback
import joblib
import numpy as np
import pandas as pd
from chefboost import Chefboost as chef
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/", methods=['GET'])
def hello():
    return "hey"


@app.route('/predict', methods=['POST'])
def predict():
    lr = chef.load_model("model.pkl")
    if lr:
        try:
            json = request.get_json()
            model_columns = joblib.load("model_cols.pkl")
            temp = list(json[0].values())
            vals = np.array(temp)
            prediction = chef.predict(lr, temp)
            print("here:", prediction)
            return jsonify({'prediction': str(prediction[0])})

        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        return ('No model here to use')


if __name__ == '__main__':
    app.run(debug=True)
