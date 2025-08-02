# SCT_ML_3
This project implements a classic machine learning pipeline to classify images of cats and dogs. It uses the Histogram of Oriented Gradients (HOG) for feature extraction and a Support Vector Machine (SVM) for classification.

The script performs the following key steps:
1. Load Data: It reads image files from a dataset directory, which should contain separate subfolders for 'cat' and 'dog' images.
2. Preprocess Images: Each image is resized to a fixed 64x64 size and converted to grayscale.
3. Feature Extraction: It calculates HOG features for each image. HOG is effective at capturing shape and texture information, which helps distinguish between different objects.
4. Train the Model: The dataset is split into training and testing sets. An SVM classifier is then trained on the HOG features from the training data.
5. Evaluate Performance: The trained model's accuracy is tested on the unseen test set, and a detailed classification report is printed to show its performance.

You will need Python 3 and the following libraries:
1. Opencv-python
2. Numpy
3. Scikit-image
4. Scikit-learn
