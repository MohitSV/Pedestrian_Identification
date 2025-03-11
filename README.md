# Pedestrian Detection 🚶‍♂️ 👀

This project implements two different approaches for pedestrian detection:

## 1. Haar Cascade Classifier 🎯
A machine learning based approach that uses cascade of simple features (Haar-like features) to detect objects. It works by:
- Using rectangular regions at specific locations in a detection window
- Summing up pixel intensities in each region
- Calculating the difference between these sums
- Training multiple stages of classifiers for robust detection

## 2. HOG (Histogram of Oriented Gradients) + SVM 📊
A feature descriptor-based method combined with Support Vector Machine classifier:
- HOG calculates the distribution of gradient directions in local regions
- Creates a histogram of gradient orientations in localized portions of an image
- SVM uses these HOG features to classify whether a given image window contains a pedestrian

Both methods are well-established in computer vision for pedestrian detection, with HOG+SVM generally providing better accuracy but requiring more computational resources. 🔍
