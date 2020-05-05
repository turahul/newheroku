from flask_ngrok import run_with_ngrok
from flask import Flask,render_template , request
import tensorflow as tf
import dlib
import cv2
import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from matplotlib import pyplot as imshow
model = load_model('/content/drive/My Drive/Colab Notebooks/deepfake2.h5')
np.set_printoptions(suppress=True)

def predict(filename):
  input_shape = (128, 128, 3)
  pr_data = []
  m=0
  n=0
  detector = dlib.get_frontal_face_detector()
  cap = cv2.VideoCapture('/content/drive/My Drive/Colab Notebooks/uploads/'+filename)
  frameRate = cap.get(5)
  while cap.isOpened():
    frameId = cap.get(1)
    ret, frame = cap.read()
    if ret != True:
      break
    if frameId % ((int(frameRate)+1)*1) == 0:
      face_rects, scores, idx = detector.run(frame, 0)
      for i, d in enumerate(face_rects):
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()
        crop_img = frame[y1:y2, x1:x2]
        crop_img = cv2.resize(crop_img, (224, 224))
        #data = data.reshape(-1, 128, 128, 3)
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image_array = np.asarray(crop_img)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1         
        data[0] = normalized_image_array
        prediction = model.predict(data)
        if prediction[0][0] > 0.50:
          m=m+1
        else:
          n=n+1
        if m > n:
          pred="Fake"
          return "Fake"
        else:
          pred= "Real"
          return "Real"

  



app = Flask(__name__,template_folder="/content/drive/My Drive/Colab Notebooks/templates")
run_with_ngrok(app)   #starts ngrok when the app is run
@app.route("/", methods = ["GET","POST"])
def home():
  if request.method=="POST":
    file = request.files["file"]
    file.save(os.path.join("/content/drive/My Drive/Colab Notebooks/uploads",file.filename))
    filename = file.filename
    #predict(filename)
    return render_template("WebPage3.html", message=predict(filename))
        
        
  return render_template("WebPage3.html", message="upload")
  
app.run()