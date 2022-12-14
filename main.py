# Import libraries
import os

import cv2
import matplotlib.pyplot as plt


# Define paths
data_dir = './data'
models_dir = './models'

# Load models
face_detector = cv2.CascadeClassifier(os.path.join(models_dir, 'haarcascade_frontalface_default.xml'))

# Load images
for img_path in os.listdir(data_dir):
    img = cv2.imread(os.path.join(data_dir, img_path))
    #convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_detector.detectMultiScale(img_gray, 1.3, 5)
    plt.figure()
    for face in faces:
        x1, y1, w, h = face
        # Draw rectangle around face 
        img = cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (0, 255, 0), 5)
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
    plt.imshow(img_rgb)
        
plt.show()