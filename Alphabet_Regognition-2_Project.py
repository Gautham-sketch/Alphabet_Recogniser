import cv2
from cv2 import cvtColor
import pandas as pd
import numpy as np
import PIL.ImageOps
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

x = np.load('Alphabet_Regognition.npz')
y = pd.read_csv("Alphabet_Regognition.csv")
classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
nclasses = len(classes)

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=2500,train_size=7500)
xtrain_scaled = xtrain/255.0
xtest_scaled = xtest/255.0

lr = LogisticRegression(solver="saga", multi_class="multinomial").fit(xtrain_scaled,ytrain)
y_predict = lr.predict(xtest_scaled)
accuracy = accuracy_score(ytest,y_predict)
print("Accuracy is :- ", accuracy)

capture = cv2.VideoCapture(0)
while(True):
    tf, frame = capture.read()
    gray = cvtColor(frame, cv2.COLOR_BGR2GRAY)
    width, height = gray.shape()
    north_west = (int(width/2-75), int(height/2-75))
    south_east = (int(width/2+75), int(height/2+75))
    cv2.rectangle(gray,north_west,south_east,(0,0,255),3)
    roi = gray[north_west[1]:south_east[1], north_west[0]:south_east[0]]
    focus_image = Image.fromarray(roi)
    image_bw = focus_image.convert('L')
    image_bw_resized = image_bw.resize((28,28),Image.ANTIALIAS)
    image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted - min_pixel,0,255)
    max_pixel = np.max(image_bw_resized_inverted)
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    test_pred = lr.predict(test_sample)
    print("Predicted class is :- ",test_pred)
    cv2.imshow("Alphabet recogniser",gray)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
capture.release()
cv2.destroyAllWindows()