import cv2
import numpy as np
from nms import *

import glob2 as gl
import cv2 as cv
import imutils   
import numpy as np

import matplotlib.pyplot as plt
from sklearn import metrics

# Initializing the HOG person detector

hog = cv.HOGDescriptor() 
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

true_path = gl.glob("C:\Abhi\Documents\College material\Semester 5\Image Processing\Project\Test_True\*.png")
false_path = gl.glob("C:\Abhi\Documents\College material\Semester 5\Image Processing\Project\Test_False\*")

true_target = [1 for i in range(len(true_path))]
false_target = [0 for i in range(len(false_path))]
target = true_target + false_target

rects = list()
prediction = list()
true_pos = 0
true_neg= 0
false_neg = 0
false_pos = 0

def appendRects(i, j, conf, c, rects):
    x = int((j)*pow(scaleFactor, c))
    y = int((i)*pow(scaleFactor, c))
    w = int((64)*pow(scaleFactor, c))
    h = int((128)*pow(scaleFactor, c))
    rects.append((x, y, conf, w, h))
    
def performNMS(img):
    nms_rects = nms_HOG(rects, 0.2)

    for (a, b, c, d) in nms_rects:
        cv2.rectangle(img, (a, b), (a+c, b+d), (0, 255, 0), 2)

'''
    The following function is used to analyse the positive and negative cases
    by separately running the test cases through the loop and returning a list
    which consists of the number of detected cases as well as the number of
    cases that were not detected.
'''

def runHOGAnalysis(image_path):
    detect = 0
    no_detect = 0
    
    for img in image_path:
        image = cv.imread(img)
        
        # Resizing the Image 
        image = imutils.resize(image, width=min(400, image.shape[1])) 
           
        # Detecting all the regions in the image that has a pedestrians inside it 

        (regions, _) = hog.detectMultiScale(image, winStride=(3, 3), padding=(4, 4), scale=1.05)
         
        '''
             'regions' variable contains an array containing all the regions detected by the algorithm

             If we want to find out whether the function has detected or not, we can just check if
             the regions variable is an array or not. If nothing has been detected, it returns a
             tuple containing float values
        '''
        
        if type(regions) != np.ndarray:
            no_detect += 1
            prediction.append(0)

        else:
            detect += 1
            prediction.append(1)
            for box in regions:
                rects.append(list(box))

            # Drawing the regions in the Image 
            for (x, y, w, h) in regions: 
                cv.rectangle(image, (x, y),  
                              (x + w, y + h),  
                              (0, 0, 255), 2)
                
        performNMS(image)
        
    return [detect, no_detect]


true_cases = runHOGAnalysis(true_path)
false_cases = runHOGAnalysis(false_path)


true_pos = true_cases[0]
false_neg = true_cases[1]

false_pos = false_cases[0]
true_neg = false_cases[1]

precision = true_pos/(true_pos + false_pos)
recall = true_pos/(true_pos + false_neg)

f1_score = (2*precision*recall)/(precision+recall)

report = metrics.classification_report(target, prediction, output_dict = True)

print("Number of True Positive images: ", true_pos,
      "\nNumber of True Negative images: ", true_neg,
      "\nNumber of False Positive images: ", false_pos,
      "\nNumber of False Negative images: ", false_neg,
      "\nPrecision: %0.6f" %precision,
      "\nRecall: %0.6f" %recall, 
      "\nF1-score: %0.6f" %f1_score,
      "\nAccuracy of HOG: %0.3f" % (report["accuracy"] * 100), "%")
print("\n\n")
print(metrics.classification_report(target, prediction))

cv.destroyAllWindows()

# Plotting the histogram for the different performance values

# To save the new image
        
# cv2.imwrite("../Output_images/output.jpg", img)






