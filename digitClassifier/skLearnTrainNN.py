from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import cv2

digits = np.array(pd.read_csv("c:\\users\\devar\\downloads\\mnist_train.csv"))

x1 = digits[0, 1:]
#print(sum(x1))
x1.shape = (28, 28)
cv2.imshow("img", x1)
cv2.waitKey(0)
    


x = digits[:30000,  1:]
y = digits[:30000, :1]

lm = LogisticRegression()
print("Model training...in progress....")
lm.fit(x/255, y)
# The model is now trained. We could split the data into training and testing parts, and check for the accuracy. We can also use the model to predict from our own image. We will use the opencv module to read and manipulate the image.

    # In[ ]:

joblib.dump(lm, "digitNN.cpickle")
print("Model is trained")