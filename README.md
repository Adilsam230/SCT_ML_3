# SCT_ML_3
This project implements a classic machine learning pipeline to classify images of cats and dogs. It uses Support Vector Machine (SVM) for classification.

The script performs the following key steps:
1. Loads the Data
2. Processes the Images: For each image, it converts it to grayscale, resizes it to a uniform 64x64 pixels, and then "flattens" the image into a single, long list of its pixel values.
3. Trains the model: The script uses a classic machine learning model, a Support Vector Machine (SVM), and trains it on 80% of the image data. A StandardScaler is used to normalize the pixel data, which is a crucial step for the SVM to perform well.
4. Evaluate: Finally, it tests the trained model on the remaining 20% of the images to see how accurately it can predict whether an unseen image is a cat or a dog. The final accuracy and a detailed classification report are printed as the output.

You will need Python 3 and the following libraries:
1. Opencv-python
2. Numpy
3. Scikit-learn
