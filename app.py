from flask import Flask,render_template , request
import os
import tensorflow as tf
import dlib
import cv2
import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
model = load_model('fullset.h5')

app = Flask(__name__,template_folder="templates")


@app.route('/' , methods = ["GET","POST"] )

def home():
    if request.method=="POST":
        file = request.files["file"]
        file.save(os.path.join("uploads",file.filename))
        filename = str(file.filename)
        
        detector = dlib.get_frontal_face_detector()
        cap = cv2.VideoCapture('uploads/'+ filename)
        np.set_printoptions(suppress=True)
        m=0
        n=0
        inc=0
        frameRate = cap.get(5)
        while cap.isOpened() and inc < 12:
          frameId = cap.get(1)
          ret, frame = cap.read()
          if ret != True:
            break
          if frameId % ((int(frameRate)+1)*1) ==  and inc < 12:
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
              print(prediction)
              inc=inc+1
              if prediction[0][0] > 0.50:
                print("fake")
                m=m+1
               
              else:
                print("real")
                n=n+1
                
        if m > n :
          m=int((m/(m+n))*100)
	       
          x = str(m)
          predict="Fake Video"+"\t" + x + "  %"
          return render_template("index.html",message=predict,per=x)
        else:
          n=int((n/(m+n))*100)
          
          y = str(n)
          predict="Real Video"+ "\t" + y + "  %"
          return render_template("index.html",message=predict,per=y)         

        
    return render_template("index.html", message="upload")

if __name__ == "__main__":
	#decide what port to run the app in
	port = int(os.environ.get('PORT', 5000))
	#run the app locally on the givn port
	app.run(host='0.0.0.0', port=port)
	#optional if we want to run in debugging mode
	#app.run(debug=True)

