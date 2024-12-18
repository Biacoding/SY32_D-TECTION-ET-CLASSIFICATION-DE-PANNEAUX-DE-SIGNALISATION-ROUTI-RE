import os
import csv
import pandas as pd
import numpy as np
import cv2
from skimage import transform,feature
from sklearn.preprocessing import LabelEncoder
import random
import joblib
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, average_precision_score,recall_score
#%% Data augmentation

# Horizontal Flip
def random_horizontal_flip(image, p=0.5):
    return cv2.flip(image, 1)
 
# Color augmentation
def random_color_jitter(image, brightness=0.2, contrast=0.2, saturation=0.2):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    
    v = v * (1 + random.uniform(-brightness, brightness))
    v = np.clip(v, 0, 255).astype(hsv_image.dtype)

    s = s * (1 + saturation)
    s = np.clip(s, 0, 255).astype(hsv_image.dtype)

    hsv_image = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

def augment_image(image):
    image = random_horizontal_flip(image)
    image = random_color_jitter(image)
    return image


#%% feature extraction
def extract_color_histogram(image, image_size=(64, 64),bins=(8, 8, 8)):
    resized_image = cv2.resize(image, image_size)
    hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_hog_features(image, image_size=(64, 64)):
    resized_image = cv2.resize(image, image_size)  # Resize image to ensure consistent size
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    hog_features, hog_image = feature.hog(gray, orientations=9, pixels_per_cell=(8, 8),
                                          cells_per_block=(2, 2), visualize=True)
    return hog_features

#%% load train data
def load_data(dataset,image_folder, label_folder,background_folder, image_size=(64, 64), ignore_label='ff'):
    features = []
    labels = []
    csv_empty = []
    num_ff = 0
    size_err = 0
    fail_label = 0
    label_empty = 0
    image_files = sorted(os.listdir(image_folder))
    print('num of image total : ', len(image_files))
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        label_file = image_file.replace('.jpg', '.csv')  
        label_path = os.path.join(label_folder, label_file)
        
        if os.path.exists(label_path):
            try:
                with open(label_path, newline='') as csvfile:
                    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                    for row in reader:
                        if len(row) == 0:
                            label_empty += 1
                            csv_empty.append(csvfile)
                            continue
                        row = row[0].split(',')
                        if len(row) < 5:
                            continue
                        label_class = row[4]
                        if label_class == ignore_label:
                            continue
                        if label_class=='background':
                            num_ff = num_ff + 1
                        try:
                            x_min = int(row[0])
                            y_min = int(row[1])
                            x_max = int(row[2])
                            y_max = int(row[3])
                            
                            cropped_image = image[y_min:y_max, x_min:x_max]
                            resized_image = transform.resize(cropped_image, image_size, anti_aliasing=True)
                            hog_features = extract_hog_features(cropped_image, image_size=image_size)
                            #color_histogram = extract_color_histogram(cropped_image,image_size=image_size)
                            pixel_features = resized_image.flatten()
                            
                            # weight
                            hog_weight = len(pixel_features) / len(hog_features)
                            #color_weight = len(hog_features) / len(color_histogram)
                            
                            weighted_hog_features = hog_features * hog_weight
                            
                            #feature = np.hstack([color_features, pixel_features])
                            feature = np.hstack([pixel_features,weighted_hog_features])
                           
                                
                            if len(features) > 0 and len(feature) != len(features[0]):
                                size_err = size_err + 1
                                continue
                            
                            features.append(feature)
                            labels.append(label_class)
                            
                            # data augmentation
                            au_image = augment_image(cropped_image)
                            au_resized_image = transform.resize(au_image, image_size, anti_aliasing=True)
                            au_hog_features = extract_hog_features(au_image, image_size=image_size)
                            #color_histogram = extract_color_histogram(cropped_image,image_size=image_size)
                            au_pixel_features = au_resized_image.flatten()
                            
                            # weight
                            au_hog_weight = len(au_pixel_features) / len(au_hog_features)
                            #color_weight = len(hog_features) / len(color_histogram)
                            
                            au_weighted_hog_features = au_hog_features * au_hog_weight
                            
                            au_feature = np.hstack([au_pixel_features,au_weighted_hog_features])
                           
                            if len(features) > 0 and len(au_feature) != len(features[0]):
                                size_err = size_err + 1
                                continue
                            
                            features.append(au_feature)
                            labels.append(label_class)
                            
                        except ValueError:
                            continue
            except pd.errors.EmptyDataError:
                fail_label += 1
                pass
            
    # add background in train set
    if dataset == 'train':
        background_files = sorted(os.listdir(background_folder))
        for background_file in background_files:
            background_path = os.path.join(background_folder, background_file)
            background = cv2.imread(background_path)
            if background is None:
                continue
            resized_background = transform.resize(background, image_size, anti_aliasing=True)
            hog_features_background = extract_hog_features(background, image_size=image_size)
            #color_histogramb = extract_color_histogram(background, image_size=image_size)
            pixel_features_background = resized_background.flatten()
            
            hog_weight = len(pixel_features_background) / len(hog_features_background)
            #color_weight = len(hog_features_background) / len(color_histogramb)
            weighted_hog_features_background = hog_features_background * hog_weight
            
            feature_background = np.hstack([pixel_features_background,weighted_hog_features_background])
            features.append(feature_background)
            labels.append('background')
        
    X = np.array(features)
    y = np.array(labels)

    print('fail_label : ', fail_label)
    print('label_empty :', label_empty)
    print('size_err :', size_err)
    print('num_ff :',num_ff)
    return X, y

# Define the paths for training and testing sets
train_image_folder = 'train/images'
train_label_folder = 'train/labels'
test_image_folder = 'val/images'
test_label_folder = 'val/labels'
negative_samples_folder = "negative_samples"

#%% prepare data set
# Training & validation set
X_train, y_train = load_data('train',train_image_folder, train_label_folder,negative_samples_folder, image_size=(64, 64), ignore_label='ff')

X_test, y_test = load_data('test',test_image_folder, test_label_folder,negative_samples_folder, image_size=(64, 64), ignore_label='ff')

if X_train.shape[0] != len(y_train):
    print(f"Warning: Training data and labels have inconsistent lengths: {X_train.shape[0]} and {len(y_train)}")

if X_test.shape[0] != len(y_test):
    print(f"Warning: Test data and labels have inconsistent lengths: {X_test.shape[0]} and {len(y_test)}")

# defining a custom label order
custom_labels = ['ceder', 'danger', 'forange', 'frouge', 'fvert', 'interdiction', 'obligation', 'stop','background']

label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(custom_labels)
# Label Encoding
y_train_encoded = label_encoder.transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

label_mapping = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))

# Clean data to remove NaN values
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

print(f'Training set size: {X_train.shape}, {y_train.shape}')

#%% train model SVM

# Initialize the SVM classifier
svm_classifier = SVC(probability=True, random_state=42)

# Use encoded labels for training
svm_classifier.fit(X_train, y_train_encoded)

# Make predictions on the test set
y_test_pred = svm_classifier.predict(X_test)
y_test_proba = svm_classifier.predict_proba(X_test)

# Calculate the accuracy of the test set
accuracy_test = accuracy_score(y_test_encoded, y_test_pred)
print("Val set Accuracy: {:.2f}%".format(accuracy_test * 100))

recall_test = recall_score(y_test_encoded, y_test_pred,average='micro')
print("Val set Recall: {:.2f}%".format(recall_test * 100))

# Calculate the confusion matrix
conf_matrix_test = confusion_matrix(y_test_encoded, y_test_pred)
print("Val set Confusion Matrix:\n", conf_matrix_test)

# Print Classification Report
class_report_test = classification_report(y_test_encoded, y_test_pred, zero_division=1)
print("Val set Classification Report:\n", class_report_test)

# Calculate mAP for test set
n_classes = len(np.unique(y_train_encoded))
aps_test = []
for i in range(n_classes):
    y_true = (y_test_encoded == i).astype(int)
    y_scores = y_test_proba[:, i]
    if np.sum(y_true) == 0:
        print(f'Class {i} has no positive samples in y_true. Skipping AP calculation.')
        continue
    ap = average_precision_score(y_true, y_scores)
    aps_test.append(ap)
    print(f'Class {i} AP: {ap}')

mAP_test = np.mean(aps_test)
print('Val set Mean Average Precision (mAP):', mAP_test)

#%% save SVM model
joblib.dump(svm_classifier, 'svm_classifier_model_back_067.joblib')