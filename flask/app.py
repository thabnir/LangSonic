from flask import Flask, render_template, request, redirect
import tensorflow as tf
import mp3tospect
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model("cnn.h5")
if model is not None:
    model.test_on_batch(np.zeros((1, 13, 250, 1)), np.zeros((1, 5)))


def do_tests():
    # get mp3 files in data at
    mp3path = "../data/mp3/"
    langs = ["de", "en", "es", "fr", "it"]
    paths = [mp3path + lang + "/" for lang in langs]
    # get first 20 files from each
    paths = [path + file for path in paths for file in os.listdir(path)[:20]]
    # get the predictions
    predictions = [predict_from_mp3(path) for path in paths]
    [print(path) for path in paths]


do_tests()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return render_template("index.html", error="No file part")

    file = request.files["file"]
    print(f"Got file: `{file}`")

    if file.filename == "":
        return render_template("index.html", error="No selected file")

    if file:
        filepath = f"./data/{file.filename}"
        print(filepath)
        file.save(filepath)
        prediction = predict_from_mp3(filepath)
        # convert to a list of tuples, where each tuple is (language, probability)
        prediction = list(zip(langs, prediction))
        # sort them by probability
        print(prediction)
        prediction.sort(key=lambda x: x[1], reverse=True)
        return render_template("results.html", prediction=prediction)

    # TODO: handle error
    print("Something went wrong")
    return render_template("index.html", error="Something went wrong")


def predict_from_mp3(path):
    """
    Args:
        path: path to the mp3 file

    Returns:
        list containing the probabilities for each language
    """
    return model.predict(np.array([mp3tospect.model_input_from_audio(path)])).tolist()[
        0
    ]


from languages import LANGUAGES

langs = ["German", "English", "Spanish", "French", "Italian", "Arabic", "Hindi"]


def get_language(prediction):
    return LANGUAGES[langs[np.argmax(prediction)]]


if __name__ == "__main__":
    app.run(debug=True)
