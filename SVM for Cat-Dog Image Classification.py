import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

IMAGE_SIZE = (64, 64)
DATASET_DIR = r"C:\Users\adils\Downloads\train\train"
CATEGORIES = ['cat', 'dog']
SAMPLE_SIZE = 1000
def load_and_process_data(directory, categories, max_samples):
    features_list = []
    labels_list = []
    for label, category in enumerate(categories):
        folder_path = os.path.join(directory, category)

        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found for '{category}', skipping.")
            continue

        print(f"Processing '{category}' images...")

        for filename in os.listdir(folder_path)[:max_samples]:
            image_path = os.path.join(folder_path, filename)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                img_resized = cv2.resize(img, IMAGE_SIZE)
                features_list.append(img_resized.flatten())
                labels_list.append(label)

    return np.array(features_list), np.array(labels_list)

if __name__ == "__main__":
    X_data, y_data = load_and_process_data(DATASET_DIR, CATEGORIES, SAMPLE_SIZE)

    if X_data.size == 0:
        print("Error: No data loaded. Check your dataset path.")

    else:
        X_scaled = StandardScaler().fit_transform(X_data)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_data, test_size=0.2, random_state=42, stratify=y_data
        )
        print(f"\nTraining SVM model...")
        model = svm.SVC(class_weight='balanced').fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("\n--- Evaluation Results ---")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=CATEGORIES, zero_division=0))