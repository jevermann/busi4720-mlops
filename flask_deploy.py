import keras
import flask
from flask import request
import pandas as pd

# Load the trained model
norm_boston_model = keras.saving.load_model('norm.boston.model.trained.save')

# Load the form and info page
with open("predict_form.html") as f:
    form_html = f.read()
with open("info_page.html") as f:
    info_html = f.read()
# Load the form and info page
with open("predict_form_async.html") as f:
    form_async_html = f.read()

# A predict function for the model
def predict(inputs):
    return norm_boston_model.predict_on_batch(inputs)[0][0]

app = flask.Flask(__name__)

@app.route("/form", methods=["GET"])
def say_form():
    return form_html

@app.route("/form_async", methods=["GET"])
def say_form_async():
    return form_async_html

@app.route("/", methods=["GET"])
def say_info():
    return info_html

@app.route("/predict_json", methods=["POST"])
def predict_json():
    reply = {}
    # TODO: Input checking goes here
    # TODO: Input logging goes here
    inputs = pd.DataFrame.from_dict(request.json).transpose()
    prediction = predict(inputs)
    # TODO: Output checking goes here
    # TODO: Output logging goes here
    reply["prediction"] = str(prediction)
    reply["success"] = True
    return flask.jsonify(reply)

@app.route("/predict_form", methods=["POST"])
def predict_form():
    # TODO: Input checking goes here
    # TODO: Input logging goes here
    inputs = pd.DataFrame([float(i) for i in request.form.to_dict().values()]).transpose()
    prediction = predict(inputs)
    # TODO: Output checking goes here
    # TODO: Output logging goes here
    return flask.render_template("predict_result.html", prediction=str(prediction))

app.run()
