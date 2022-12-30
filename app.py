import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import sklearn

model = pickle.load(open('model.pkl','rb'))
dataset = pd.read_csv('diabetes.csv')

dataset_X = dataset.iloc[:,[1, 2, 5, 7]].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_X)
app=Flask(__name__)

@app.route('/')

def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    # # pregnancies = request.form.get('Pregnancies')
    # glucose = request.form.get('Glucose')
    # # bloodpressure = request.form.get('BloodPressure')
    # # skinthickness = request.form.get('SkinThickness')
    # insulin = request.form.get('Insulin')
    # bmi = request.form.get('BMI')
    # # diabetespedigreefunction = request.form.get('DiabetesPedigreeFunction')
    # age = request.form.get('Age')

    # #prediction

    # result=model.predict(np.array([glucose,insulin,bmi,age]).reshape(1,4))
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict( sc.transform(final_features) )

    if prediction == 1:
        pred = "You have Diabetes, please consult a Doctor."
    elif prediction == 0:
        pred = "You don't have Diabetes."
    output = pred

    return render_template('index.html', prediction_text='{}'.format(output))


if __name__=='__main__':
    app.run(debug=True)