# PROJECT: DETECTION AND CLASSIFICATION OF ROAD SIGNS
### SY32 TD1_Projet1_GroupeF
**Beijia Mao & Branchu Corentin**


## Environment Configuration

In order to run this project, you need to configure your Python environment first. The following steps will guide you through `requirements.txt` to install all the required dependencies.

### Install dependencies with pip

Open your command line interface and run the following command to install the dependency:

```bash
pip install -r requirements.txt
```

Ensure that the `requirements.txt` file is located in the current directory, or provide the full path to the file.

### Install dependencies with conda

If you are using Anaconda or Miniconda, you can create a new environment and install all required dependencies:

```bash
conda create --name sy32 --file requirements.txt
```

After installation, activate the environment with the following command:

```bash
conda activate sy32
```

## Project Overview

This project consists of two main parts: Machine Learning (ML) and Deep Learning (DL).

### Machine Learning Part

The machine learning section includes the following three scripts, which use traditional machine learning techniques, such as Support Vector Machines (SVM) and feature extraction (like HOG), for image recognition tasks:

- `ML_HOG_SVM.py`：Uses HOG features and SVM for image classification.
- `ML_detect_dynamique_sliding_window.py`：Implements a dynamic sliding window method to identify objects in images.
- `ML_detect_selective_search.py`：Utilizes selective search to enhance the accuracy of object detection.

However, we recommend using `ML_HOG_SVM.py` + `ML_detect_selective_search.py`, dynamic sliding windows take a lot of time to recognize, which is not efficient

- `svm_classifier_model.joblib`：Already trained model

### Deep Learning Part

The deep learning section contains the following three scripts, which are based on the deep learning framework PyTorch and use convolutional neural networks (like ResNet) for complex image recognition tasks:

- `DL_ResNet_two_steps.py`：Uses the ResNet model for two-step image recognition, first determining whether an image region contains an object, then classifying the object.
- `DL_detect_dynamique_sliding_window.py`：Combines dynamic sliding window technology and deep learning models for object recognition in images.
- `DL_detect_selective_search.py`：Applies deep learning models and selective search techniques for more precise image analysis.

We recommend using `DL_ResNet_two_steps.py` + `DL_detect_selective_search.py` for the same reason.

- `dl_binaire_part1_final.pth`：Already trained binary model 
- `dl_binaire_part2_final.pth`：Already trained classification model

Notably, in the deep learning part, we utilize two networks for target recognition as well as target classification, respectively.

## Closing Words

Thank you for your interest in this project. Please feel free to contact the project developers with any questions and suggestions.

**Beijia Mao** ：beijia.mao@etu.utc.fr

**Branchu Corentin** ：corentin.branchu@etu.utc.fr
