from flask import Flask, render_template, request

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
        file.save(file.filename)
        return render_template("index.html", message="File uploaded successfully")

    return render_template("index.html", error="Something went wrong")


if __name__ == "__main__":
    app.run(debug=True)
