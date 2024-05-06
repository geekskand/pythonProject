from flask import Flask, render_template, request, jsonify, Response
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import base64

app = Flask(__name__)

# Load the machine learning model
classifier = Classifier(r"C:\Users\kantr\PycharmProjects\signlang\Model\keras_model.h5",r"C:\Users\kantr\PycharmProjects\signlang\Model\labels.txt")
classifier.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

def gen_frames():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    offset = 20
    imgSize = 300
    labels = ["A", "B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","X","Y","Z"]
    while True:
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            imgCropShape = imgCrop.shape
            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                print(prediction, index)
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
            cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                          (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x-offset, y-offset),
                          (x + w+offset, y + h+offset), (255, 0, 255), 4)
        ret, buffer = cv2.imencode('.jpg', imgOutput)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/predict', methods=['POST'])
# @app.route('/predict', methods=['POST'])
def predict():
    image_data = request.get_json()['image']
    image_array = np.frombuffer(base64.b64decode(image_data), np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Pre-process the image
    img = cv2.resize(img, (224, 224))
    img = img / 255.0

    # Make predictions using the sign language model
    prediction = classifier.model.predict(img[np.newaxis,...])
    index = np.argmax(prediction[0])  # Get the index of the maximum value in the first row

    # Get the corresponding letter from the labels list
    labels = ["A", "B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","X","Y","Z"]
    letter = labels[index]
    print(prediction)

    return jsonify({'prediction': letter})

if __name__ == "__main__":
    app.run(debug=True)