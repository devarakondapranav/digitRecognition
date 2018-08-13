from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import cv2
from sklearn.externals import joblib
from PIL import Image, ImageFilter


from PIL import Image, ImageFilter


def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.fromarray(np.uint8((argv)*255))
    
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (0)) #creates white canvas of 28x28 pixels
    
    if width > height: #check which dimension is bigger
        #Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0/width*height),0)) #resize height according to ratio width
        if (nheight == 0): #rare case but minimum is 1 pixel
            nheight = 1  
        # resize and sharpen
        img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight)/2),0)) #caculate horizontal pozition
        newImage.paste(img, (4, wtop)) #paste resized image on white canvas
    else:
        #Height is bigger. Heigth becomes 20 pixels. 
        nwidth = int(round((20.0/height*width),0)) #resize width according to ratio height
        if (nwidth == 0): #rare case but minimum is 1 pixel
            nwidth = 1
         # resize and sharpen
        img = im.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth)/2),0)) #caculate vertical pozition
        newImage.paste(img, (wleft, 4)) #paste resized image on white canvas
    
    #newImage.save("sample.png")
    

    tv = list(newImage.getdata()) #get pixel values
    
    #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [ (255-x)*1.0/255.0 for x in tv] 
    
    tva = np.array(tva)
    tva = 1 - tva
    tva.shape = (28, 28)
    
    tva.shape = (1, 784)
    return tva


    #print(tva)

def digitClassifier(path):
    #path = center_extent(path, (28, 28))
    path = imageprepare(path)

    
    '''
    digits = np.array(pd.read_csv("c:\\users\\devar\\downloads\\mnist_train.csv"))
    


    x = digits[:2000,  1:]

    y = digits[:2000, :1]
    print(x.shape)
    print(y.shape)


    lm = LogisticRegression()

    print("Model training...")
    lm.fit(x, y)
    '''
    lm = joblib.load("C:\\Users\\devar\\Documents\\Python programs\\opencv\\digitClassifier\\digitNN.cpickle")


    # The model is now trained. We could split the data into training and testing parts, and check for the accuracy. We can also use the model to predict from our own image. We will use the opencv module to read and manipulate the image.

    # In[ ]:

    '''
    
    filename = path
    W = 28.
    oriimg = path
    height, width= oriimg.shape
    imgScale = W/width
    newX,newY = oriimg.shape[1]*imgScale, oriimg.shape[0]*imgScale
    newimg = cv2.resize(oriimg,(int(newX),int(newY)))
    #gray = cv2.cvtColor(newimg, cv2.COLOR_BGR2GRAY)

    gray = newimg
    
    gray = 255-gray
    gray.resize(28, 28)
    cv2.imshow("gray", gray)
    cv2.waitKey(0)
    
    gray.shape = (1,784)
    #print(gray)
    '''
    


    # The above code does the following
    #     1). Read an image. This image could be an image of any digit, with a white background and the digit itslef written in black
    #     2). Resize the image appropriately. Here we resize the image to a size of 8pixels X 8pixels(since the size of images in the         dataset is 8X8.
    #     3). Convert the image to grayscale to make sure we are not dealing with RGB values.
    #     4). Since the background is white in color and the digit in black, we reverse these colors to match with those in the               dataset
    #     5). The object gray now is an 8X8 numpy array that matches the images in the dataset. We use this `gray` object to predict           what digit it is.

    # In[ ]:


    y1 = lm.predict(path)
    return(y1[0])
    font = cv2.FONT_HERSHEY_SIMPLEX
    name = str(y1)
    color = (255, 255, 0)
    stroke = 2
    displayImg = cv2.imread(filename)
    cv2.putText(displayImg, name, (28, 28), font, 1, color, stroke, cv2.LINE_AA)
    #cv2.imshow("img", displayImg)
    #cv2.waitKey()
    #print(y1)



'''
imageArray = cv2.imread("c:\\users\\devar\\documents\\8.jpg")
imageArray = cv2.cvtColor(imageArray, cv2.COLOR_BGR2GRAY)
print(digitClassifier(imageArray))
'''


