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
    pregnancies = int(request.form.get('pregnancies'))
    glucose = int(request.form.get('glucose'))
    bloodpressure = int(request.form.get('bloodpressure'))
    skinthickness = int(request.form.get('skinthickness'))
    insulin =int( request.form.get('insulin'))
    bmi =float( request.form.get('bmi'))
    diabetespedigreefunction = float(request.form.get('diabetespedigreefunction'))
    age = int(request.form.get('age'))

    #prediction

    result=model.predict(np.array([pregnancies,glucose,bloodpressure,skinthickness,insulin,bmi,diabetespedigreefunction,age]).reshape(1,8))

    return str(result)


if __name__=='__main__':
    app.run(debug=True)