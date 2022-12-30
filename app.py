from flask import Flask, request, render_template
import joblib
import sklearn  
import pickle, gzip  
import pandas as pd  
import numpy as np  
app = Flask(__name__)  
model = joblib.load('KNN HD model.pkl',"rb")  
 
 
@app.route('/')
def home():
    return render_template("home.html")

@app.route("/predict", methods=['GET','POST'])  
def predict():
    if request.method=='POST': 
        age = int(request.form["age"])  
        sex = int(request.form["sex"])  
        trestbps = int(request.form["trestbps"])
        chol = int(request.form["chol"])
        oldpeak = float(request.form["oldpeak"])
        thalach = int(request.form["thalach"])
        fbs = int(request.form["fbs"])
        exang = int(request.form["exang"])
        slope = int(request.form["slope"])
        cp = int(request.form["cp"])
        thal = int(request.form["thal"])
        ca = int(request.form["ca"])
        restecg = int(request.form["restecg"])
        
        arr = np.asarray([[age, sex, cp, trestbps,  
            chol, fbs, restecg, thalach,  
            exang, oldpeak, slope, ca,  
            thal]])
        pred = model.predict(arr)
        if (pred[0] == 0): 
            res_val = "NO HEART PROBLEM"
        else:  
            res_val = "HEART PROBLEM"  
        return render_template('home.html', prediction_text='PATIENT HAS {}'.format(res_val))  
if __name__ == "__main__":
    app.run(host='0.0.0.0' , port= 5005)  