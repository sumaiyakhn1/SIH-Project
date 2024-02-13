
from cvzone.ClassificationModule import Classifier
import cv2
import pytesseract
import time
from datetime import datetime
import tensorflow as tf
import numpy as np
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\priya\Downloads\tesseract-ocr-w64-setup-5.3.1.20230401.exe'
custom_config = r'--psm 11 --tessdata-dir C:\Users\priya\OneDrive\Desktop\Project\Model'

cap = cv2.VideoCapture(0)
maskClassifier = Classifier('C:\\Users\\priya\\OneDrive\\Desktop\\Project\\Model\\keras_model.h5', 'C:\\Users\\priya\\OneDrive\\Desktop\\Project\\Model\\labels.txt')

while True:
    _, img = cap.read()
    img_resized = cv2.resize(img, (640, 32))
    img_resized = np.reshape(img_resized, (1, 32, 640, 3))
    predection = maskClassifier.getPrediction(img)
    print(predection)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

    def extract_text_from_image(img):
      gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      extracted_text = pytesseract.image_to_string(gray_image, config='--psm 11')
      print(extracted_text)
      return extracted_text
    
    def detect_keyword(text, keyword):
      return keyword in text
    
    def detect_pothole():
      if predection == True:
        return predection
      
    def capture_image():
      cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
       cv2.imwrite("captured.jpg", frame)
       extracted_text = extract_text_from_image(frame)

    keyword = "pithole_detected" 
    print(extracted_text)
    
    if detect_keyword(extracted_text.lower(), keyword):
      if(extracted_text.lower()==keyword):
        if detect_pothole():
          start_time = time.time()
          while (time.time() - start_time) > 5:
            capture_image()
        else:
          exit()