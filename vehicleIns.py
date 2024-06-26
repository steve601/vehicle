from flask import Flask,request,render_template
import pickle
import numpy as np

app = Flask(__name__)

def load_model():
    with open('vehicle.pkl','rb') as file:
        data = pickle.load(file)
    return data

objects = load_model()
model = objects['model']
scaler = objects['scaler']

@app.route('/')
def homepage():
    return render_template('vehicle.html')

@app.route('/predict',methods = ['POST'])
def do_prediction():
    a = request.form.get('age')
    b = request.form.get('previously_insured')
    c = request.form.get('vehicle_age')
    d = request.form.get('vehicle_damage')
    e = request.form.get('policy_sales_channel')
    
    if c == '> 2 Years':
        c = 0
    if c == '< 1 Year':
        c =1
    if c == '1-2 Year':
        c = 2
    
    d = 0 if d == 'No' else 1
    
    x = np.array([[a,b,c,d,e]])
    x = scaler.transform(x)
    prediction = model.predict(x)
    
    msg = 'Customer might be intrested' if prediction == 1 else 'Customer might not be intrested'
    
    return render_template('vehicle.html',text=msg)

if __name__ == '__main__':
    app.run(host = "0.0.0.0")