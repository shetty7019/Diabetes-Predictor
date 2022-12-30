import pickle
from flask import Flask,render_template,request

import numpy as np

model = pickle.load(open('model.pkl','rb'))

app=Flask(__name__)

@app.route('/')

def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict_diabetes():
    # pregnancies = request.form.get('Pregnancies')
    glucose = request.form.get('Glucose')
    # bloodpressure = request.form.get('BloodPressure')
    # skinthickness = request.form.get('SkinThickness')
    insulin = request.form.get('Insulin')
    bmi = request.form.get('BMI')
    # diabetespedigreefunction = request.form.get('DiabetesPedigreeFunction')
    age = request.form.get('Age')

    #prediction

    result=model.predict(np.array([glucose,insulin,bmi,age]).reshape(1,4))

    if result[0]==1:
     result='Diabetic'
    else:
     result='Not Diabetic'
    return render_template('index.html',result=result)


if __name__=='__main__':
    app.run(debug=True)