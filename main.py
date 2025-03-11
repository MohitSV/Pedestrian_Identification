import cv2
import numpy as np
from nms import *

import glob2 as gl
import cv2 as cv
import imutils
import matplotlib.pyplot as plt
from sklearn import metrics

pedsCascade =  cv.CascadeClassifier("haarcascade_pedestrian.xml")
hog         = cv.HOGDescriptor() 

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

def runAnalysis(image_path, class_type):
    detect = 0
    no_detect = 0
    
    for img in image_path:
        
        image = cv.imread(img)
        
        # Resizing the Image 
           
        # Detecting all the regions in the image that has a pedestrians inside it 
        
        if class_type == "hog":
            image = imutils.resize(image, width=min(400, image.shape[1]))
            (regions, _) = hog.detectMultiScale(image, winStride=(3, 3), padding=(4, 4), scale=1.05)
            
        elif class_type == "cascade":
            regions = pedsCascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=9, minSize=(30, 30))
         
        else:
            print("Error! No such classifier exists!")
            return []
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
        
            performNMS(image)
                
    return [detect, no_detect]

def performance(true_cases, false_cases):
    true_pos = true_cases[0]
    false_neg = true_cases[1]
    
    false_pos = false_cases[0]
    true_neg = false_cases[1]
    
    precision = (true_pos/(true_pos + false_pos)) * 100
    recall = (true_pos/(true_pos + false_neg)) * 100
    
    f1_score = (2*precision*recall)/(precision+recall)
    
    accuracy = (true_pos + true_neg)/(true_pos + true_neg + false_neg + false_pos) * 100
    
    # report = metrics.classification_report(target, prediction, output_dict = True)
    
    print("Number of True Positive images: ", true_pos,
          "\nNumber of True Negative images: ", true_neg,
          "\nNumber of False Positive images: ", false_pos,
          "\nNumber of False Negative images: ", false_neg,
          "\nPrecision: %0.6f" %precision, "%",
          "\nRecall: %0.6f" %recall, "%", 
          "\nF1-score: %0.6f" %f1_score,
          
          "\nAccuracy of : %0.3f" % accuracy, "%")      #(report["accuracy"] * 100)
    print("\n\n")
    
    return [precision, recall, f1_score, accuracy]
    # print(metrics.classification_report(target, prediction))
    
    # del report
    
    
true_cases_HOG = runAnalysis(true_path, class_type = "hog")
false_cases_HOG = runAnalysis(false_path, class_type = "hog")

prediction = list()

true_cases_Cascade = runAnalysis(true_path, class_type = "cascade")
false_cases_Cascade = runAnalysis(false_path, class_type = "cascade")

HOG_results = performance(true_cases_HOG, false_cases_HOG)
Cascade_results = performance(true_cases_Cascade, false_cases_Cascade)


fig2 = plt.figure()
plt.title("Precision")
ax2 = fig2.add_axes([0, 0, 1, 1])
Method_used = ['Cascade', 'HOG']
precision = [Cascade_results[0], HOG_results[0]]
ax2.bar(Method_used, precision)
plt.show()

fig3 = plt.figure()
plt.title("Recall")
ax3 = fig3.add_axes([0, 0, 1, 1])
Method_used = ['Cascade', 'HOG']
recall = [Cascade_results[1], HOG_results[1]]
ax3.bar(Method_used, recall)
plt.show()

fig1 = plt.figure()
plt.title("Accuracy")
ax1 = fig1.add_axes([0, 0, 1, 1])
Method_used = ['Cascade', 'HOG']
accuracy = [Cascade_results[3], HOG_results[3]]
ax1.bar(Method_used, accuracy)
plt.show()


















