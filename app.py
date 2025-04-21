from flask import Flask, request, jsonify,render_template # type: ignore
import joblib # type: ignore
import numpy as np # type: ignore

app = Flask(__name__)

# Load the model
model = joblib.load('health.joblib')

@app.route('/')
def home():     
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form.get("age"))
    bmi = float(request.form.get("bmi"))
    children = int(request.form.get("children"))
    region = request.form.get("region")
    if(region== "southwest"):
        region = 0
    elif(region == "southeast"):
        region = 1
    elif(region == "northwest"):
        region = 2
    else:
        region = 3
    gender = request.form.get("gender")
    if(gender=="Male"):
        gender = 1
    else:
        gender = 0
    smoker = request.form.get("smoker")
    if(smoker=="Yes"):
        smoker = 1
    else:
        smoker = 0
    # You can customize this based on expected input
    features = np.array([age,gender,bmi,children,smoker,region]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return render_template("index.html",prediction=prediction)
@app.route("/header")
def header():
    return render_template("header.html")


if __name__ == '__main__':
    app.run(debug=True)
