from flask import Flask, render_template, request
import tensorflow as tf

app = Flask(__name__)


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
        # Perform any necessary processing on the file (e.g., save it, analyze it, etc.)
        # For simplicity, this example just saves the file in the same directory.
        # todo:
        file.save(file.filename)
        return render_template("index.html", message="File uploaded successfully")

    return render_template("index.html", error="Something went wrong")


def gen(video):
    while True:
        frame = video.get_frame()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")


model = tf.keras.models.load_model(".h5")


class Model:
    def preprocess(img):
        return tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255).flow(
            img, batch_size=32, shuffle=True
        )


if __name__ == "__main__":
    app.run(debug=True)
