import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
from sklearn.preprocessing import normalize
import argparse

# Define the path to your trained model
MODEL_PATH = 'face_recognition_resnet50.pth'

# Load the trained model and class names
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load checkpoint with weights_only=True to avoid potential security issues
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)

# Define the model
model = models.resnet50(weights=None)  # Load model without pre-trained weights
num_ftrs = model.fc.in_features
model.fc = torch.nn.Identity()  # Remove the classification layer
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)  # Ensure model is on the same device
model.eval()

# Load class names
class_names = checkpoint['class_names']

# Define the transforms for prediction
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Define face detection and feature extraction functions
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_features(image):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert('RGB')
    img = data_transforms(img).unsqueeze(0)  # Add batch dimension
    img = img.to(device)  # Ensure the input tensor is on the same device as the model

    with torch.no_grad():
        features = model(img)
    
    return features.cpu().numpy().flatten()

def compute_distance(features1, features2):
    features1 = normalize([features1])
    features2 = normalize([features2])
    return np.linalg.norm(features1 - features2)

# Dummy database of known features (to be replaced with actual database)
known_features = np.random.rand(len(class_names), 2048)  # Adjust size if needed

def annotate_image(image_path, output_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        features = extract_features(face_img)
        
        distances = [compute_distance(features, known_feature) for known_feature in known_features]
        min_index = np.argmin(distances)
        label = class_names[min_index]
        
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    cv2.imwrite(output_path, img)

def main():
    parser = argparse.ArgumentParser(description="Face recognition and annotation script")
    parser.add_argument("input_image_path", type=str, help="Path to the input image")
    parser.add_argument("output_image_path", type=str, help="Path to save the annotated image")
    args = parser.parse_args()
    
    annotate_image(args.input_image_path, args.output_image_path)

if __name__ == "__main__":
    main()
