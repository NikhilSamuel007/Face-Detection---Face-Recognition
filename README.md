# Face-Detection---Face-Recognition
Face Detection using OpenCV &amp; Face Recognition using ResNet50 model

This project aims to train a face recognition model using a pre-trained ResNet50 network. The script allows for training, validation, and saving of the model
# Face Recognition
## Table of Contents
  Requirements
  Installation
  Dataset Structure
  Usage
  Training
  Saving the Model

## Requirements
## Requirements

- Python
- PyTorch
- torchvision
- OpenCV
- Pillow
- scikit-learn
- numpy
- OpenCV

## Installation
Clone the repository:
git clone https://github.com/NikhilSamuel007/Face-Detection-and-Face-Recognition/face_recognition_resnet50.git
cd face_recognition_resnet50

## Dataset Structure
Your dataset should be organized in the following structure:

dataset/
    train/
        class1/
            image1.jpg
            image2.jpg
            ...
        class2/
            image1.jpg
            image2.jpg
            ...
        ...
    val/
        class1/
            image1.jpg
            image2.jpg
            ...
        class2/
            image1.jpg
            image2.jpg
            ...
        ...

## Training
To train the model, run the training.py script with the required arguments:
python training_final.py --data_dir "path/to/dataset" --num_epochs 100 --batch_size 32 --learning_rate 0.001

## Arguments
- --data_dir: Path to the dataset directory.
- --num_epochs: Number of epochs for training (default: 25).
- --batch_size: Batch size for training (default: 32).
- --learning_rate: Learning rate for the optimizer (default: 0.001).

## Saving the Model
After training, the script saves the best model to face_recognition_resnet50.pth, which includes the model state dictionary and class names.

## Prediction
python face_recognition_detection.py /path/to/input/image.jpg /path/to/output/image.jpg

## Results:
![unknown_face_image](https://github.com/user-attachments/assets/75854126-db15-49b7-b214-89c03c8c6b67)

# Face Detection

This project performs face detection on a set of images using a Haar Cascade classifier. It also calculates performance metrics for the detected faces against ground truth annotations.

## Features
- Detects faces in images using the Haar Cascade classifier.
- Draws bounding boxes around detected faces.
- Saves images with detected faces to a specified folder.
- Calculates performance metrics (precision, recall, F1-score) and IoU.
- Measures and reports average detection time per image.

## Requirements
Python
OpenCV
pandas
scikit-learn

To run the face detection and calculate performance metrics, use the following command:
python face_detection.py --image_folder <image_folder_path> --pred_folder <pred_folder_path> --ground_truth_path <ground_truth_csv_path>

## Arguments
- --image_folder: Path to the folder containing images.
- --pred_folder: Path to the folder to save predicted images.
- --ground_truth_path: Path to the CSV file containing ground truth annotations.

Ground Truth Annotations
The ground truth annotations CSV file should have the following columns:

image: The filename of the image.
xmin: The x-coordinate of the top-left corner of the bounding box.
ymin: The y-coordinate of the top-left corner of the bounding box.
xmax: The x-coordinate of the bottom-right corner of the bounding box.
ymax: The y-coordinate of the bottom-right corner of the bounding box.

## Output
The script prints the following performance metrics to the console:
- Total images processed
- Total faces detected
- Average detection time per image
- Precision
- Recall
- F1 Score
- Accuracy
