from copyreg import pickle
import numpy as np
from flask import Flask, render_template, url_for, request, jsonify
import pickle
import model as m

app = Flask(__name__)


@app.route('/', methods = ['GET','POST'])
def index():
    if request.method == "POST":
        gender = request.form["Gender"]
        age = request.form["Age"]
        hypertension = request.form["Hypertension"]
        heart = request.form["HeartDisease"]
        married = request.form["Marriage"]
        work = request.form["WorkType"]
        residence = request.form["Residence"]
        glucose = request.form["GlucoseLevel"]
        bmi = request.form["BMI"]
        smoking = request.form["Smoking"]
        #stroke_predicted = []
        stroke_predicted = m.stroke_prediction(age,hypertension,heart,married,residence,glucose,bmi,gender,work,smoking)
        sp = stroke_predicted

    return render_template('index.html', my_stroke = sp)

# @app.route('/sub', methods = ['POST'])
# def submit():
#     if request.method == "POST":
#         gender = request.form["Gender"]
#         age = request.form["Age"]
#         hypertension = request.form["Hypertension"]
#     return render_template("sub.html", g = gender, a = age)

if __name__== "__main__":
    app.run(debug=True)