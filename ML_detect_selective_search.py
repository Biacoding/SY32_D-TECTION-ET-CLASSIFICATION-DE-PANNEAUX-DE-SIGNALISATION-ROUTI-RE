import os
import cv2
import numpy as np
from skimage import transform, feature
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import joblib
import time
import selectivesearch
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='skimage.feature.texture')

custom_labels = ['ceder', 'danger', 'forange', 'frouge', 'fvert', 'interdiction', 'obligation', 'stop', 'background']
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(custom_labels)

# Load the trained model
svm_classifier = joblib.load('svm_classifier_model.joblib')
# image folder
test_image_folder = 'dataset-main-test/test'
# augument data
def augment_image(image, brightness=0.2, saturation=0.2):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    
    # Adjusting Brightness
    v = v * (1 + brightness)
    v = np.clip(v, 0, 255).astype(hsv_image.dtype)

    # Adjust saturation
    s = s * (1 + saturation)
    s = np.clip(s, 0, 255).astype(hsv_image.dtype)

    hsv_image = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

def draw_prediction(image, box, labels_probs, color=(0, 255, 0)):
    x_min, y_min, x_max, y_max = box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
    for i, (label, prob) in enumerate(labels_probs):
        text = f"{label} ({prob:.2f})"
        cv2.putText(image, text, (x_min, y_min - 10 - i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def load_test_images(image_folder):
    image_files = sorted(os.listdir(image_folder))
    images = []
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        if image is not None:
            images.append((image_file, image))
    return images

def extract_color_histogram(image, image_size=(64, 64), bins=(8, 8, 8)):
    resized_image = cv2.resize(image, image_size)
    hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_hog_features(image, image_size=(64, 64)):
    resized_image = cv2.resize(image, image_size)  # Resize image to ensure consistent size
    resized_image = (resized_image * 255).astype(np.uint8)  # Convert to 8-bit
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = (gray * 255).astype(np.uint8)
    hog_features, hog_image = feature.hog(gray, orientations=9, pixels_per_cell=(8, 8),
                                          cells_per_block=(2, 2), visualize=True)
    return hog_features

def iou(boxA, boxB):
    boxA = list(map(int, boxA))  # Make sure the coordinates are integers
    boxB = list(map(int, boxB))
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxB[3], boxB[3])
    
    if xB < xA or yB < yA:
        return 0.0

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def selective_search(image, scale=500, sigma=0.9, min_size=10, min_region_size=2000, aspect_ratio_threshold=1.2):
    #  Execute the Selective Search algorithm
    img_lbl, regions = selectivesearch.selective_search(image, scale=scale, sigma=sigma, min_size=min_size)
    
    # Create a collection for storing candidate regions
    candidates = set()
    
    for r in regions:
        # Excluding duplicate candidate areas
        if r['rect'] in candidates:
            continue
        # Excluding candidate areas smaller than a given size
        if r['size'] < min_region_size:
            continue
        # Get the coordinates and dimensions of the candidate area
        x, y, w, h = r['rect']
        # Exclusion of candidate areas with non-compliant aspect ratios
        if w / h > aspect_ratio_threshold or h / w > aspect_ratio_threshold:
            continue
        # Adding a candidate region to a collection
        candidates.add(r['rect'])
    
    return candidates

def filtre_nms(results_in, tresh_iou=0.5):
    filtered_results = []
    for result in results_in:
        if result[6][0][0] != 'background':
            filtered_results.append(result)
    results_in = np.array(filtered_results)
    results_out = np.empty((0, 7), dtype=object)  
    unique_ids = np.unique(results_in[:, 0])
    for i in unique_ids:  
        results_in_i = results_in[results_in[:, 0] == i]
        results_in_i = results_in_i[results_in_i[:, 5].argsort()[::-1]]
        results_out_i = np.empty((0, 7), dtype=object)  
        results_out_i = np.vstack((results_out_i, results_in_i[0]))
        
        for n in range(1, len(results_in_i)):
            for m in range(len(results_out_i)):
                if iou(results_in_i[n, 1:5], results_out_i[m, 1:5]) > tresh_iou:
                    break
                elif m == len(results_out_i) - 1:
                    results_out_i = np.vstack((results_out_i, results_in_i[n]))
        
        results_out = np.vstack((results_out, results_out_i))
    
    return results_out

# Load test images
test_images = load_test_images(test_image_folder)

# Using Selective Search
image_size = (64, 64)  
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
all_results = []
for image_file, image in test_images:
    image_id = int(image_file.split('.')[0])
    original_image = image.copy()
    print('Predicting: ', image_id)
    
    candidates = selective_search(image, scale=300, sigma=0.8, min_size=50, min_region_size=800, aspect_ratio_threshold=5)
    
    for (x, y, w, h) in candidates:
        window = image[y:y + h, x:x + w]
        
        # Extraction Features
        window = augment_image(window)
        resized_window = transform.resize(window, image_size, anti_aliasing=True)
        pixel_features = resized_window.flatten()  
        #color_histogram = extract_color_histogram(window, image_size=(64, 64))
        hog_features = extract_hog_features(window, image_size=(64, 64))
        
        hog_weight = len(pixel_features) / len(hog_features)
        weighted_hog_features = hog_features * hog_weight
        # Splicing Features
        img_feature = np.hstack([pixel_features, weighted_hog_features])
        
        # predict
        probs_window = svm_classifier.predict_proba(img_feature.reshape(1, -1))[0]
        
        # select and keep top 3 classes & prob
        top_indices = np.argsort(probs_window)[-3:][::-1]
        labels_probs = [(label_encoder.classes_[index], probs_window[index]) for index in top_indices]
        
        if labels_probs[0][1] > 0.75:
            x_min = x
            y_min = y
            x_max = x + w
            y_max = y + h
            all_results.append([image_id, x_min, y_min, x_max, y_max, labels_probs[0][1], labels_probs])

print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))             

# Make sure all_results is a 2D array
all_results = np.array(all_results, dtype=object)
if all_results.ndim == 1:
    all_results = all_results.reshape(-1, 7)

# Apply Non-Maximum Suppression (NMS)
nms_results = filtre_nms(all_results, tresh_iou=0.05)

# Plot results
output_dir = 'result'
for image_file, image in test_images:
    image_id = int(image_file.split('.')[0])
    original_image = image.copy()
    
    results_for_image = nms_results[nms_results[:, 0].astype(int) == image_id]
    for result in results_for_image:
        _, x_min, y_min, x_max, y_max, _, labels_probs = result
        draw_prediction(original_image, (int(x_min), int(y_min), int(x_max), int(y_max)), labels_probs)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Predictions for {image_file}")
    plt.axis('off')
    plt.show()
#%%
# Save results to CSV
import csv
output_csv = 'ml_detection_ss.csv'
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    for result in nms_results:
        image_id, x_min, y_min, x_max, y_max, score, labels_probs = result
        label = labels_probs[0][0]
        writer.writerow([image_id, x_min, y_min, x_max, y_max, score, label])

print(f"Detection results saved to {output_csv}")
