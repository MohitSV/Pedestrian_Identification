import cv2 as cv
# import sys
import imutils
import glob2 as gl
import numpy as np

from sklearn import metrics

# imagePath = sys.argv[1]

true_path = gl.glob("C:\Abhi\Documents\College material\Semester 5\Image Processing\Project\Test_True\*")
false_path = gl.glob("C:\Abhi\Documents\College material\Semester 5\Image Processing\Project\Test_False\*")


cascPath = "haarcascade_pedestrian.xml"
# cascPath = "pedestrian_another.xml"
# cascPath = "haarcascade_fullbody.xml"

pedsCascade =  cv.CascadeClassifier(cascPath)

true_target = [1 for i in range(len(true_path))]
false_target = [0 for i in range(len(false_path))]
target = true_target + false_target

rects = list()
prediction = list()
true_pos = 0
true_neg= 0
false_neg = 0
false_pos = 0



def runCascadeAnalysis(image_path):
    detect = 0
    no_detect = 0
    
    for img in image_path:
        
        image = cv.imread(img)
        
        # Resizing the Image 
        # image = imutils.resize(image, width=min(400, image.shape[1])) 
           
        # Detecting all the regions in the image that has a pedestrians inside it 

        regions = pedsCascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=9, minSize=(30, 30))
         
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
                
    return [detect, no_detect]
                
true_cases = runCascadeAnalysis(true_path)
false_cases = runCascadeAnalysis(false_path)

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
      "\nAccuracy of Cascade Classifier: ", report["accuracy"])
print("\n\n")
print(metrics.classification_report(target, prediction))

cv.destroyAllWindows()








