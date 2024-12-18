import torch
import os
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import csv
import torch.nn.functional as F
import matplotlib.pyplot as plt
import selectivesearch
from DL_ResNet_two_steps import ResNet34

LABELS = ['ceder', 'danger', 'forange', 'frouge', 'fvert', 'interdiction', 'obligation', 'stop', 'background']

binary_model_path = 'dl_binaire_part1_final.pth'
binary_model = ResNet34(2)  
binary_model.load_state_dict(torch.load(binary_model_path))
binary_model.eval().cuda()

classification_model_path = 'dl_binaire_part2_final.pth'
classification_model = ResNet34(num_classes=len(LABELS))  
classification_model.load_state_dict(torch.load(classification_model_path))
classification_model.eval().cuda()

test_image_folder = 'dataset-main-test/test'

transform = T.Compose([
    T.Resize((128, 128)),  
    T.ToTensor(),  
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])
#%% fonctions basic
def load_test_images(image_folder):
    image_files = sorted(os.listdir(image_folder))
    images = []
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        if image is not None:
            images.append((image_file, image))
    return images

def selective_search(image, scale=500, sigma=0.9, min_size=10, min_region_size=2000, aspect_ratio_threshold=1.2):
    # Execute the Selective Search algorithm
    img_lbl, regions = selectivesearch.selective_search(image, scale=scale, sigma=sigma, min_size=min_size)
    
    # Create a collection for storing candidate regions
    candidates = set()
    for r in regions:
        # Excluding Duplicate Candidate Areas
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

def draw_prediction(image, box, prob, label, color=(0, 255, 0)):
    x_min, y_min, x_max, y_max = box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
    text = f"{label} ({prob:.2f})"
    cv2.putText(image, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        

def iou(boxA, boxB):
    boxA = list(map(int, boxA))  # Make sure the coordinates are integers
    boxB = list(map(int, boxB))
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    if xB < xA or yB < yA:
        return 0.0

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    # Preventing too great a difference in the area of two box with too great an overlap
    if interArea >= 0.5*boxAArea or interArea >= 0.5*boxBArea :
        iou = 1
    return iou

def filtre_nms(results_in, tresh_iou=0.5):
    filtered_results = []
    for result in results_in:
        if result[6] != 'background':
            filtered_results.append(result)
    results_in = np.array(filtered_results,dtype=object)
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
#%% trait_image + selective_search
def trait_image(image_id, original_image, all_results):
    candidates = selective_search(image, scale=300, sigma=0.8, min_size=50, min_region_size=1000, aspect_ratio_threshold=4)
    
    for (x, y, w, h) in candidates:
        window = original_image[y:y + h, x:x + w]
        
        # Convert windows to PIL images for preprocess
        window = Image.fromarray(window)  
        window = transform(window)
        window = window.unsqueeze(0).cuda()  # Add batch dimension and move to GPU
        # use binary_model
        with torch.no_grad():
            prediction_b = binary_model(window)
            probabilities = F.softmax(prediction_b, dim=1)
            if probabilities[0, 1] < 0.98:  # Index 1 corresponds to the positive class, prob < 0.8 => background
                continue
        # use classification_model 
        with torch.no_grad():
            prediction = classification_model(window)
            prediction = F.softmax(prediction, dim=1)  
            max_prob, max_index = torch.max(prediction, dim=1) # Select the type of maximum probability
            max_prob = max_prob.cpu().numpy()[0]
            max_index = max_index.cpu().numpy()[0]
        
        if max_prob > 0.8:
            x_min = x
            y_min = y
            x_max = x + w
            y_max = y + h
            all_results.append([image_id, x_min, y_min, x_max, y_max, max_prob, LABELS[max_index]])
    
    return all_results

#%% detection
test_images = load_test_images(test_image_folder)
all_results = []
for image_file, image in test_images:
    image_id = int(image_file.split('.')[0])
    original_image = image.copy()
    all_results = trait_image(image_id,original_image,all_results)
    print('image:',image_id,'detected')

# Make sure all_results is a 2D array
all_results = np.array(all_results, dtype=object)
if all_results.ndim == 1:
   all_results = all_results.reshape(-1, 7)
print(all_results.shape)
#%% Removing overlapping boxes with NMS
nms_results = filtre_nms(all_results, tresh_iou=0.1)  
 
#%% plot results
for image_file, image in test_images:
    image_id = int(image_file.split('.')[0])
    original_image = image.copy()
    
    results_for_image = nms_results[nms_results[:, 0].astype(int) == image_id]
    for result in results_for_image:
        _, x_min, y_min, x_max, y_max, prob, label = result
        draw_prediction(original_image, (int(x_min), int(y_min), int(x_max), int(y_max)),prob, label)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Predictions for {image_file}")
    plt.axis('off')
    
    # Save the figure as a PNG file
    # output_filename = f"result/predictions_{image_id}.jpg"
    # plt.savefig(output_filename)
   
    plt.show()
#%% Enregistrer les résultats des tests dans un fichier CSV
output_csv = 'detection_dl_new5.csv'
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    for result in nms_results:
        image_id, x_min, y_min, x_max, y_max, prob, label = result
        writer.writerow([image_id, x_min, y_min, x_max, y_max, prob, label])

print(f"Detection results saved to {output_csv}")