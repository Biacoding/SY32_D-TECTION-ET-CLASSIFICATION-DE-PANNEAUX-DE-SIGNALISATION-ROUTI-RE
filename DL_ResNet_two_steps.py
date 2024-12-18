import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import csv
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

# labels
LABELS = ['ceder', 'danger', 'forange', 'frouge', 'fvert', 'interdiction', 'obligation', 'stop', 'background']

# Creating binary tags
BINARY_LABELS = {label: (0 if label == 'background' else 1) for label in LABELS}

# image_folder
train_image_folder = 'train/images'
train_label_folder = 'train/labels'
test_image_folder = 'val/images'
test_label_folder = 'val/labels'
negative_samples_folder = "negative_samples"

#%% BasicBlock & ResNet34FCN
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet34(nn.Module):
    def __init__(self, num_classes=9):
        super(ResNet34, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

#%% load train data
def load_data(dataset, image_folder, label_folder, background_folder, image_size=(128, 128), ignore_label='ff'):
    features = []
    labels = []
    labels_binaire = []  # for binary model
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
                        if label_class == 'background':
                            num_ff = num_ff + 1
                        try:
                            x_min = int(row[0])
                            y_min = int(row[1])
                            x_max = int(row[2])
                            y_max = int(row[3])
                            
                            cropped_image = image[y_min:y_max, x_min:x_max]
                            cropped_image = cv2.resize(cropped_image, image_size)
                            
                            feature = cropped_image
                            
                            features.append(feature)
                            labels.append(LABELS.index(label_class))
                            # binaire
                            labels_binaire.append(BINARY_LABELS[label_class]) # cible 
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
            background = cv2.resize(background, image_size)
            features.append(background)
            labels.append(LABELS.index('background')) # background
            
            labels_binaire.append(BINARY_LABELS['background']) 

    X = np.array(features)
    y = np.array(labels)
    z = np.array(labels_binaire)
    print('fail_label : ', fail_label)
    print('label_empty :', label_empty)
    print('size_err :', size_err)
    print('num_ff :', num_ff)
    return X, y, z

#%% ArrayDataset
class ArrayDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]
        image = Image.fromarray(image) 
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long) 
        return image, label

transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#%% train_model fonction
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, train Loss: {epoch_loss:.4f}')

        # val
        model.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []
        for images, labels in val_loader:
            images = images.cuda()
            labels = labels.cuda()
            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        print(f'Validation Loss: {val_loss:.4f}')
        
        # Calcul des matrices de confusion et d'autres indicateurs statistiques
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        conf_matrix = confusion_matrix(all_targets, all_predictions)
        class_accuracy_total = accuracy_score(all_targets, all_predictions)
        class_recall_total = recall_score(all_targets, all_predictions, average=None, zero_division=0)
        class_accuracy = accuracy_score(all_targets, all_predictions)
        class_recall = recall_score(all_targets, all_predictions, average=None, zero_division=0)
        print("Confusion Matrix:\n", conf_matrix)
        print("Accuracy total: ", class_accuracy_total)
        print("Recall total: ", class_recall_total)
        print("Accuracy per class: ", class_accuracy)
        print("Recall per class: ", class_recall)
#%% filter_target_images fonctions
# Filtering target images
def filter_target_images(features, labels, labels_binaire):
    X_train_filtered = []
    y_train_filtered = []

    for feature, label, label_binaire in zip(features, labels, labels_binaire):
            if label_binaire == 1:
                X_train_filtered.append(feature)
                y_train_filtered.append(label)

    return X_train_filtered, y_train_filtered

#%% main fonction
def main():
    # Initialization of models, loss function and optimizer
    n_classes = len(LABELS)
    binary_model = ResNet34(2).cuda()                  # Value judgment if image is target, not background
    classification_model = ResNet34(n_classes).cuda()  # Value judgment image is which type

    binary_criterion = nn.CrossEntropyLoss()
    classification_criterion = nn.CrossEntropyLoss()

    binary_optimizer = optim.Adam(binary_model.parameters(), lr=0.0001)
    classification_optimizer = optim.Adam(classification_model.parameters(), lr=0.0001)

    # load data sat
    X_train, y_train, z_train = load_data('train', train_image_folder, train_label_folder, negative_samples_folder, image_size=(128, 128), ignore_label='ff')
    X_val, y_val, z_val = load_data('val', test_image_folder, test_label_folder, negative_samples_folder, image_size=(128, 128), ignore_label='ff')

    # binary model dataset
    train_dataset_b = ArrayDataset(X_train, z_train, transform=transform)
    val_dataset_b = ArrayDataset(X_val, z_val, transform=transform)

    train_loader_b = DataLoader(train_dataset_b, batch_size=32, shuffle=True)
    val_loader_b = DataLoader(val_dataset_b, batch_size=32, shuffle=False)
    
    # classification model dataset
    X_train_filtered, y_train_filtered = filter_target_images(X_train,y_train,z_train)
    X_val_filtered, y_val_filtered = filter_target_images(X_val, y_val, z_val)

    filtered_train_dataset = ArrayDataset(X_train_filtered, y_train_filtered, transform=transform)
    filtered_val_dataset = ArrayDataset(X_val_filtered, y_val_filtered, transform=transform)

    filtered_train_loader = DataLoader(filtered_train_dataset, batch_size=32, shuffle=True)
    filtered_val_loader = DataLoader(filtered_val_dataset, batch_size=32, shuffle=False)
    
    # Train the first net
    train_model(binary_model, train_loader_b, val_loader_b, binary_criterion, binary_optimizer, num_epochs=15) #25
    
    # Train the secon net
    train_model(classification_model, filtered_train_loader, filtered_val_loader, classification_criterion, classification_optimizer, num_epochs=25) #50
#%%
    # # save binary model
    torch.save(binary_model.state_dict(), 'dl_binaire_part1_final.pth')
    # # save classification model
    torch.save(classification_model.state_dict(), 'dl_binaire_part2_final.pth')
#%%
if __name__ == "__main__":
    main()
