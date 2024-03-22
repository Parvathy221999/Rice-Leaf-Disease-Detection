import numpy as np
from keras.preprocessing import image
import tensorflow as tf
from flask import Flask, request
from io import BytesIO

model = tf.keras.models.load_model("rice_leaf_diseases_classification_model.h5")

class_names = ["Bacterial leaf blight", 'Brown spot', 'Leaf smut',"unrelated_image"]

def predict_image(image_bytes):
    img = image.load_img(BytesIO(image_bytes), target_size=(220, 220))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=-1)
    predicted_class_name = class_names[predicted_class_index[0]]
    return predicted_class_name

app = Flask(__name__)

@app.route("/rice_leaf", methods=["POST"])
def predict_rice_leaf():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded"
        file = request.files["file"]
        if file.filename == "":
            return "No file selected"
        try:
            image_bytes = file.read()
            predicted_class = predict_image(image_bytes)
            return ({"predicted_image":predicted_class})
        except Exception as e:
            return str(e)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8080)
