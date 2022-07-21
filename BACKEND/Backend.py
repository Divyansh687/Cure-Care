from flask import Flask,render_template,request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
classifier = pickle.load(open('heart_dis.pkl', 'rb'))
model = pickle.load(open('liver_dis.pkl', 'rb'))
from sklearn.ensemble import RandomForestClassifier
app = Flask(__name__)
@app.route('/')
def front_page():
    return render_template('result.html')
@app.route('/heart',methods=['GET','POST'])
def heart_page():
    if request.method == 'GET':
        return render_template('heart.html')
    else:
        age = request.form['age']
        sex = request.form['sex']
        chest = request.form['chest']
        trestbps = request.form['trestbps']
        chol = request.form['chol']
        fbs = request.form['fbs']
        restecg = request.form['restecg']
        thalach = request.form['thalach']
        exang = request.form['exang']
        oldpeak = request.form['oldpeak']
        heart_dataset = pd.read_csv('heart.csv')
        X = heart_dataset.drop(['slope','ca','thal','target'], axis=1) 
        y = heart_dataset['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        classifier = LogisticRegression()
        classifier.fit(X_train, y_train)
        input_data = (age,sex,chest,trestbps,chol,fbs,restecg,thalach,exang,oldpeak)
        input_data_as_numpy_array= np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        prediction = classifier.predict(input_data_reshaped)
        senddata=""
        print(prediction)
        if (prediction[0]== 0):
            senddata='According to the given details person does not have Heart Disease'
        else:
            senddata='According to the given details chances of having Heart Disease are High, So Please Consult a Doctor'
        return render_template('result.html',resultvalue=senddata)

@app.route('/liver',methods=['GET','POST'])
def liver_page():
    if request.method == 'GET':
        return render_template('liver.html')
    else:
        age = request.form['age']
        sex = request.form['sex']
        total_bilirubin = request.form['bilirubin']
        alkaline_phosphotase = request.form['alkaline']
        alamine_aminotransferase = request.form['alamime']
        aspartate_aminotransferase = request.form['asparate']
        total_proteins = request.form['proteins']
        albumin = request.form['albumin']
        albumin_and_globulin_ratio = request.form['ratio']
        liver_dataset = pd.read_csv('indian_liver_patient.csv')
        liver_dataset['Gender'] = liver_dataset['Gender'].map({'Male': 1, 'Female': 2})
        liver_dataset.dropna(inplace=True)
        indep_data = liver_dataset.drop(['Direct_Bilirubin','Dataset'], axis=1)
        dep_data = liver_dataset['Dataset']
        indep_train, indep_test, dep_train, dep_test = train_test_split(indep_data, dep_data, test_size=0.4, random_state=101)
        model = RandomForestClassifier(n_estimators = 100)
        model.fit(indep_train, indep_train)
        input_data = (age,sex,total_bilirubin,alkaline_phosphotase,alamine_aminotransferase,aspartate_aminotransferase,total_proteins,albumin,albumin_and_globulin_ratio)
        input_data_as_numpy_array= np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        prediction = model.predict(input_data_reshaped)
        senddata=""
        if (prediction[0]== 2):
            senddata='According to the given details person does not have Liver Disease'
        else:
            senddata='According to the given details chances of having Liver Disease are High, So Please Consult a Doctor'
        return render_template('result.html',resultvalue=senddata)
if __name__ == '__main__':
    app.run(debug=True)
