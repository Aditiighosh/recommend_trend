from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2

app = Flask(__name__)

# Load COCO-SSD model from TensorFlow Hub
model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

def detect_clothing(image):
    # Preprocess the image
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]  # Add batch dimension

    # Perform object detection
    detections = model(input_tensor)

    # Parse detections
    detection_classes = detections["detection_classes"][0].numpy()
    detection_boxes = detections["detection_boxes"][0].numpy()
    detection_scores = detections["detection_scores"][0].numpy()

    # Filter for clothing items (IDs for shirt, pants, etc. depend on the COCO dataset)
    clothing_items = [
        {
            "class_id": int(detection_classes[i]),
            "bounding_box": detection_boxes[i].tolist(),
            "score": float(detection_scores[i]),
        }
        for i in range(len(detection_classes))
        if detection_scores[i] > 0.5  # Filter low-confidence predictions
    ]

    return clothing_items

@app.route("/process-image", methods=["POST"])
def process_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Read the uploaded image
    file = request.files["image"]
    npimg = np.fromfile(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Convert image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform detection
    clothing_items = detect_clothing(img_rgb)

    return jsonify({"clothing_items": clothing_items})

if __name__ == "__main__":
    app.run(host="localhost", port=5000)
