import os
import time
import cv2
import pandas as pd
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Function to calculate Intersection over Union (IoU)
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def main(image_folder, pred_folder, ground_truth_path):
    # Load Haar Cascade face detector
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Create the pred folder if it doesn't exist
    if not os.path.exists(pred_folder):
        os.makedirs(pred_folder)

    # Load ground truth data
    ground_truth = pd.read_csv(ground_truth_path)

    # Initialize performance metrics
    total_images = 0
    total_faces_detected = 0
    total_detection_time = 0
    ious = []

    # Lists for precision, recall, F1-score, and accuracy
    y_true = []
    y_pred = []

    # Loop through all images in the folder
    for image_name in os.listdir(image_folder):
        if image_name.endswith(('.jpg', '.png', '.jpeg')):
            # Load image
            img_path = os.path.join(image_folder, image_name)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Measure detection time
            start_time = time.time()
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            detection_time = time.time() - start_time
            
            # Update performance metrics
            total_images += 1
            total_faces_detected += len(faces)
            total_detection_time += detection_time
            
            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Save the image with detections to the pred folder
            pred_path = os.path.join(pred_folder, image_name)
            cv2.imwrite(pred_path, img)
            
            # Get ground truth bounding boxes for the current image
            gt_boxes = ground_truth[ground_truth['image'] == image_name]
            
            for _, row in gt_boxes.iterrows():
                gt_box = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
                
                max_iou = 0
                for (x, y, w, h) in faces:
                    pred_box = [x, y, x + w, y + h]
                    iou = calculate_iou(gt_box, pred_box)
                    max_iou = max(max_iou, iou)
                
                ious.append(max_iou)
                y_true.append(1)
                y_pred.append(1 if max_iou > 0.5 else 0)
            
            for (x, y, w, h) in faces:
                pred_box = [x, y, x + w, y + h]
                match_found = False
                for _, row in gt_boxes.iterrows():
                    gt_box = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
                    iou = calculate_iou(gt_box, pred_box)
                    if (iou > 0.5):
                        match_found = True
                        break
                if not match_found:
                    y_true.append(0)
                    y_pred.append(1)

    # Calculate average detection time per image
    average_detection_time = total_detection_time / total_images if total_images > 0 else 0

    # Calculate precision, recall, F1-score, and accuracy
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    # Print performance metrics
    print(f"Total images processed: {total_images}")
    print(f"Total faces detected: {total_faces_detected}")
    print(f"Average detection time per image: {average_detection_time:.4f} seconds")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Detection and Performance Metrics")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument("--pred_folder", type=str, required=True, help="Path to the folder to save predicted images.")
    parser.add_argument("--ground_truth_path", type=str, required=True, help="Path to the CSV file containing ground truth annotations.")

    args = parser.parse_args()

    main(args.image_folder, args.pred_folder, args.ground_truth_path)
