from copyreg import pickle
import numpy as np
from flask import Flask, render_template, url_for, request, jsonify
import pickle
import model as m

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index():

    word = ''

    if request.method == "POST":

        gender = float(request.form["Gender"])
        age = float(request.form["Age"])
        hypertension = float(request.form["Hypertension"])
        heart = float(request.form["HeartDisease"])
        married = float(request.form["Marriage"])
        work = float(request.form["WorkType"])
        residence = float(request.form["Residence"])
        glucose = float(request.form["GlucoseLevel"])
        bmi = float(request.form["BMI"])
        smoking = float(request.form["Smoking"])
        #stroke_predicted = []

        stroke_predicted = m.stroke_prediction(
            age, hypertension, heart, married, residence, glucose, bmi, gender, work, smoking)

        if stroke_predicted == 0:
            word = '***You are likely to NOT get a stroke***'
        elif stroke_predicted == 1:
            word = '***You are likely to get a stroke***'

        print(word)

    return render_template('index.html', my_stroke=word)
    #my_stroke = stroke_predicted

# @app.route('/sub', methods = ['POST'])
# def submit():
#     if request.method == "POST":
#         gender = request.form["Gender"]
#         age = request.form["Age"]
#         hypertension = request.form["Hypertension"]
#     return render_template("sub.html", g = gender, a = age)


if __name__ == "__main__":
    app.run(debug=True)
