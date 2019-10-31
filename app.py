from flask import Flask, request
import io
from PIL import Image
import numpy as np
from pickle import load


app = Flask(__name__)
with open("model.pickle", "rb+") as f:
    model = load(f)


@app.route("/")
def hello():
    return """
    <p>Upload photo!</p>
    <form enctype="multipart/form-data" action="/predict_digit" method="POST">
        <input type="file" name="image" accept="image/*"/>
        <input type="submit" value="Predict digit!"/>
    </form>
    """


@app.route('/predict_digit', methods=["POST", "GET"])
def predict_digit():
    image_bytes = request.files["image"].read()
    stream = io.BytesIO(image_bytes)
    image = Image.open(stream).convert("L").resize((8, 8))
    image.save("image.jpg")

    image_np = np.array(image) / 16.0
    image_np = image_np.reshape(1, 8*8).astype(np.uint8)
    image_np = np.full((1, 8*8), 16, dtype=np.uint8) - image_np
    print(image_np)
    
    prediction = model.predict(image_np)

    return "number is: " + str(prediction[0])


if __name__ == "__main__":
    app.run()