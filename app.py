from copyreg import pickle
import numpy as np
import pickle
from flask import Flask,request,jsonify, render_template


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

def round_to_closest_salary(salary):
    return int((salary+5000)/5000)*5000

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods = ['POST'])
def predict():
    input =request.form.values()
    input_int = [int(x) for x in input]
    prediction = round(model.predict([np.array(input_int)])[0],2)
    rounded_salary = round_to_closest_salary(prediction)
    return render_template('index.html',prediction_text = 'Employe Salary Should be ${}'.format(rounded_salary))

@app.route('/predict_api',methods = ['POST'])
def predict_api():
    data = request.get_jason(force = True)
    prediction = model.predict([np.array(list(data))])[0]
    return jsonify(prediction)


  

if __name__ == '__main__':
    app.run(debug = True)        
