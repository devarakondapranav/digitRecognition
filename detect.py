import numpy as np 
from digitClassifier import digitClassifier
import mahotas
import cv2

image = cv2.imread("test.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11, 11), 0)
#print("Hello world ")


edged = cv2.Canny(image, 30, 150)
'''
cv2.imshow("edged", edged)
cv2.waitKey(0)
'''
(_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in cnts], key = lambda x: x[1])
for (c, _) in cnts:
	(x, y, w, h) = cv2.boundingRect(c)
	if(w>=3 and h>=20):
		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)

		digit = gray[y:y+h, x:x+w]
		thresh = digit.copy()
		T = mahotas.thresholding.otsu(thresh)
		#print("T is " + str(T))
		thresh[thresh > T] = 255
		

		thresh = cv2.bitwise_not(thresh)
		thresh = 255-thresh
		'''
		cv2.imshow("", thresh)
		cv2.waitKey(0)
		'''



		
		#print((digit))
		answer = digitClassifier.digitClassifier(thresh)

		cv2.putText(image, str(answer), (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
		print(answer)
		#digit = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)

		cv2.imshow("edged", image)
		#cv2.waitKey(0)
cv2.waitKey(0)
print("No of objects is " + str(len(cnts)))
