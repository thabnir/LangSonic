from flask import Flask, render_template, request
import tensorflow as tf
import mp3tospect
import numpy as np
import datetime

app = Flask(__name__)

model = tf.keras.models.load_model("cnn.h5")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return render_template("index.html", error="No file part")

    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html", error="No selected file")

    if file:
        # timestamp for file name to avoid overwriting
        # timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filepath = f"./data/{file.filename}"
        print(filepath)
        file.save(filepath)
        prediction = predict_from_mp3(filepath)
        print(prediction)
        return render_template("index.html", message="File uploaded successfully")

    # TODO: handle error
    print("Something went wrong")
    return render_template("index.html", error="Something went wrong")


def predict_from_mp3(path):
    return model.predict(np.array([mp3tospect.model_input_from_mp3(path)]))


if __name__ == "__main__":
    app.run(debug=True)
