import cv2
import numpy as np
import time
from skimage.feature import hog
import joblib
from nms import nms

def appendRects(i, j, conf, c, rects):
    x = int((j)*pow(scaleFactor, c))
    y = int((i)*pow(scaleFactor, c))
    w = int((64)*pow(scaleFactor, c))
    h = int((128)*pow(scaleFactor, c))
    rects.append((x, y, conf, w, h))

clf = joblib.load("pedestrian.pkl")


# orig = cv2.imread(args["image"])
# orig = cv2.imread("C:\Abhi\Documents\College material\Semester 5\Image Processing\Project\Test_True\FudanPed00001.png")
orig = cv2.imread("C:\Abhi\Documents\College material\Semester 5\Image Processing\Project\Final files\sample_images\p2.jpg")

img = orig.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# scaleFactor = args["downscale"]
scaleFactor = 1.2
inverse = 1.0/scaleFactor
# winStride = (args["winstride"], args["winstride"])
winStride = (8, 8)
winSize = (128, 64)

rects = []

h, w = gray.shape
count = 0
while (h >= 128 and w >= 64):

    # print (gray.shape)

    h, w= gray.shape
    horiz = w - 64
    vert = h - 128
    # print (horiz, vert)
    i, j = 0, 0
    
    while i < vert:
        j = 0
        while j < horiz:

            portion = gray[i:i+winSize[0], j:j+winSize[1]]
            features = hog(portion, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2")

            result = clf.predict([features])

            if int(result[0]) == 1:
                # print (result, i, j)
                confidence = clf.decision_function([features])
                appendRects(i, j, confidence, count, rects)


            j = j + winStride[0]

        i = i + winStride[1]

    gray = cv2.resize(gray, (int(w*inverse), int(h*inverse)), interpolation=cv2.INTER_AREA)
    count = count + 1
    # print (count)

# print (rects)

nms_rects = nms(rects, 0.2)

for (a, b, conf, c, d) in rects:
    cv2.rectangle(orig, (a, b), (a+c, b+d), (0, 255, 0), 2)

cv2.imshow("Before NMS", orig)
cv2.waitKey(0)


for (a, b, conf, c, d) in nms_rects:
    cv2.rectangle(img, (a, b), (a+c, b+d), (0, 255, 0), 2)

cv2.imshow("After NMS", img)

cv2.waitKey(0)

# save output
cv2.imwrite("../output.jpg", img)



