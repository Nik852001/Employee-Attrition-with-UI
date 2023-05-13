from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
model1 = pickle.load(open('model1.pkl', 'rb'))


@app.route('/')
# @cross_origin()
def home():
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        time_spend_company = float(request.form["time_spend_company"])
        avg_montly_hours = float(request.form["avg_montly_hours"])
        number_project = float(request.form["number_project"])
        EMP_Engagement = float(request.form["EMP_Engagement"])
        Emp_Role = float(request.form["Emp_Role"])
        Percent_Remote = float(request.form["Percent_Remote"])/100
        EMP_Sat_Remote = float(request.form["EMP_Sat_Remote"])
        LinkedIn_Hits = float(request.form["LinkedIn_Hits"])

        x = pd.DataFrame({"Percent_Remote": [Percent_Remote], "number_project": [number_project],  "avg_montly_hours": [avg_montly_hours], "time_spend_company": [time_spend_company],
                           "LinkedIn_Hits": [LinkedIn_Hits],"Emp_Role": [Emp_Role], "EMP_Sat_Remote": [EMP_Sat_Remote], "EMP_Engagement": [EMP_Engagement]})
        ml = model.predict(x)

    

        if ml == 1:
            x1 = pd.DataFrame({"Percent_Remote":[Percent_Remote], "number_project":[number_project]	, "avg_montly_hours":[avg_montly_hours],
        	               "LinkedIn_Hits":[LinkedIn_Hits],	"Emp_Role":[Emp_Role],	"EMP_Sat_Remote":[EMP_Sat_Remote], 
                           "EMP_Engagement":[EMP_Engagement]})
            ml1 = model1.predict(x1)
            k = ml1-time_spend_company
        
            if k<=0:
                t="immediately"
            else:
                k = k.item()
                k = round(k, 1)
                t="within "+ str(k) + " years" 
            return render_template('index.html', result='The employee is more likely to leave the organization {}!'.format(t))
            



        else:
            g = "continue in"
            return render_template('index.html', result='The employee is more likely to {} the organization!'.format(g))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=2764)