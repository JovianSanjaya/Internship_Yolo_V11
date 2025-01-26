
# without IoU handling for one model 


# from ultralytics import YOLOv10
# import os

# # Load the custom-trained model
# weights_path = r'C:\Internship Project YoloV10\yolov10\runs\detect\train\weights\best.pt'
# model = YOLOv10(weights_path)

# # Path to the test images
# test_images_dir = r'C:\Internship Project YoloV10\dataset_real_train_valid_test_intern\test\images'
# output_dir = r'C:\Internship Project YoloV10\yolov10\runs\detect\train\results'

# # Run predictions on all images in the test directory
# results = model.predict(source=test_images_dir, save=True, project=output_dir)  # 'save=True' saves annotated images

# # Output path for the annotated images
# print(f"Annotated images saved to: {output_dir}")

# # Handle prediction results
# for result in results:
#     print(f"Image Path: {result.path}")
#     print(f"Predictions: {result.verbose()}")  # Contains detected objects and their details





#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------






















# without IoU handling for four model 

# from ultralytics import YOLOv10
# import os

# # Define paths to the model weights
# weights_paths = [
#     r'C:\Internship Project YoloV10\yolov10\runs\detect\exp15-nicks\weights\best.pt',
#     r'C:\Internship Project YoloV10\yolov10\runs\detect\exp15-dents\weights\best.pt',
#     r'C:\Internship Project YoloV10\yolov10\runs\detect\exp15-pittings\weights\best.pt',
#     r'C:\Internship Project YoloV10\yolov10\runs\detect\exp15-scratches\weights\best.pt'
# ]

# test_images_dir = r'C:\Internship Project YoloV10\dataset_real_train_valid_test_intern\test\images'
# output_dir = r'C:\Internship Project YoloV10\yolov10\runs\detect\exp15-four model personal annotation'

# # Create output directory if it does not exist
# os.makedirs(output_dir, exist_ok=True)

# # Load models
# models = [YOLOv10(weights_path) for weights_path in weights_paths]

# def combine_predictions(predictions_list):
#     combined_results = []
#     for predictions in predictions_list:
#         for result in predictions:
#             combined_results.append(result)
#     return combined_results

# def process_image_with_models(image_path, models):
#     all_predictions = []
#     image = image_path  # Initialize with the image path

#     for model in models:
#         print(f"Processing {image_path} with model {model}")
#         result = model.predict(source=image, save=False)
#         all_predictions.append(result)
        
#         intermediate_result_path = os.path.join(output_dir, f"intermediate_{os.path.basename(image_path)}")
#         for r in result:
#             r.save(intermediate_result_path)
        
#         # Update the image path for the next model
#         image = intermediate_result_path
    
#     # Combine results from all models
#     combined_results = combine_predictions(all_predictions)
#     return combined_results

# # Process all images
# for image_name in os.listdir(test_images_dir):
#     image_path = os.path.join(test_images_dir, image_name)
#     combined_results = process_image_with_models(image_path, models)

#     # Save final combined results
#     final_result_path = os.path.join(output_dir, f"final_{image_name}")
#     for result in combined_results:
#         result.save(final_result_path)
    
#     print(f"Final annotated image saved to: {final_result_path}")
#     # Additional result handling and analysis can be done here








#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------











# with IoU handling for one model 


from ultralytics import YOLO
import os
import cv2  

 # Load the custom-trained model
weights_path = r'C:\Users\jovian sanjaya putra\runs\detect\train\weights\best.pt'
#model = YOLOv10(weights_path)

model = YOLO(weights_path)

# Path to the test images
test_images_dir = r'C:\test\traditional'
output_dir = r'C:\test'


# Load the custom-trained model
# weights_path = r'C:\Internship Project YoloV10\yolov10\runs\detect\split different learning rate\train\weights\best.pt'
# model = YOLOv10(weights_path)

# # Path to the test images
# test_images_dir = r'C:\Internship Project YoloV10\dataset test not included in personal annotation'
# output_dir = r'C:\Internship Project YoloV10\yolov10\runs\detect\split different learning rate\train\weights\results_other_test_set'



# Create output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Class-specific color codes and box thicknesses
class_color_thickness = {
    'Nicks': {'color': (60, 60, 255), 'thickness': 2},     # Nicks (Red) in BGR
    'Dents': {'color': (148, 156, 255), 'thickness': 2},   # Dents (Light Red) in BGR
    'Scratches': {'color': (28, 116, 255), 'thickness': 2}, # Scratches (Orange) in BGR
    'Pittings': {'color': (28, 180, 255), 'thickness': 2}   # Pittings (Yellow) in BGR
}

# Mapping of class indices to class names
class_labels = {0: 'Nicks', 1: 'Dents', 2: 'Scratches', 3: 'Pittings'}

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def resolve_conflicts(predictions):
    final_results = []
    for pred in predictions:
        for box in pred.boxes:
            current_box = box.xyxy[0].cpu().numpy()
            current_conf = box.conf.item()
            current_cls = int(box.cls.item())

            conflict_found = False
            for final in final_results:
                final_box = final["box"]
                iou = calculate_iou(current_box, final_box)

                if iou > 0.3:  # Adjust this threshold if needed
                    conflict_found = True
                    if current_conf > final["conf"]:
                        final["box"] = current_box
                        final["conf"] = current_conf
                        final["cls"] = current_cls
                    break
            
            if not conflict_found:
                final_results.append({
                    "box": current_box,
                    "conf": current_conf,
                    "cls": current_cls
                })
    
    return final_results

# Run predictions on all images in the test directory
results = model.predict(source=test_images_dir, save=False)

# Handle prediction results
for result in results:
    image_path = result.path
    img = cv2.imread(image_path)

    # Resolve conflicts before drawing
    final_predictions = resolve_conflicts([result])

    for pred in final_predictions:
        box = pred["box"]
        conf = pred["conf"]
        class_id = pred["cls"]
        class_name = class_labels[class_id]  # Class name using the mapping

        # Get the color and thickness for the class
        color = class_color_thickness[class_name]['color']
        thickness = class_color_thickness[class_name]['thickness']

        # Draw the bounding box
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness)

        # Put the class label and confidence
        label = f"{class_name}: {conf:.2f}"
        cv2.putText(img, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    # Save the annotated image
    output_image_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_image_path, img)

    print(f"Annotated image saved to: {output_image_path}")

# Output path for the annotated images
print(f"All annotated images saved to: {output_dir}")


















# with extra euclidean distance as the measure



# from ultralytics import YOLO
# import os
# import cv2
# import numpy as np

# # Load the custom-trained model
# weights_path = r'C:\Internship Project YoloV11\ultralytics\runs\detect\others\leak valid 7%\weights\best.pt'
# model = YOLO(weights_path)

# # Path to the test images
# test_images_dir = r'C:\Internship Project YoloV11\unseen test dataset\image1.BMP'
# output_dir = r'C:\Internship Project YoloV11\unseen test dataset\leak valid 7%'

# # Create output directory if it does not exist
# os.makedirs(output_dir, exist_ok=True)

# # Class-specific color codes and box thicknesses
# class_color_thickness = {
#     'Nicks': {'color': (60, 60, 255), 'thickness': 2},     # Nicks (Red) in BGR
#     'Dents': {'color': (148, 156, 255), 'thickness': 2},   # Dents (Light Red) in BGR
#     'Scratches': {'color': (28, 116, 255), 'thickness': 2}, # Scratches (Orange) in BGR
#     'Pittings': {'color': (28, 180, 255), 'thickness': 2}   # Pittings (Yellow) in BGR
# }

# # Mapping of class indices to class names
# class_labels = {0: 'Nicks', 1: 'Dents', 2: 'Scratches', 3: 'Pittings'}

# def calculate_iou(box1, box2):
#     """Calculate the IoU (Intersection over Union) of two bounding boxes."""
#     x1 = max(box1[0], box2[0])
#     y1 = max(box1[1], box2[1])
#     x2 = min(box1[2], box2[2])
#     y2 = min(box1[3], box2[3])

#     inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
#     box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
#     box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

#     iou = inter_area / float(box1_area + box2_area - inter_area)
#     return iou

# def calculate_center_distance(box1, box2):
#     """Calculate Euclidean distance between the center points of two bounding boxes."""
#     center_x1 = (box1[0] + box1[2]) / 2
#     center_y1 = (box1[1] + box1[3]) / 2
#     center_x2 = (box2[0] + box2[2]) / 2
#     center_y2 = (box2[1] + box2[3]) / 2
#     distance = np.sqrt((center_x2 - center_x1)**2 + (center_y2 - center_y1)**2)
#     return distance

# def non_max_suppression(predictions, iou_threshold=0.1, distance_threshold=20):
#     """Apply Non-Maximum Suppression (NMS) with IoU and distance-based filtering."""
    
#     final_predictions = []

#     # Group predictions by class
#     for cls in np.unique([p['cls'] for p in predictions]):
#         class_preds = [p for p in predictions if p['cls'] == cls]
#         boxes = np.array([p['box'] for p in class_preds])
#         confidences = np.array([p['conf'] for p in class_preds])

#         indices = np.argsort(confidences)[::-1]  # Sort by confidence descending
#         keep = []

#         while len(indices) > 0:
#             current = indices[0]
#             keep.append(class_preds[current])

#             if len(indices) == 1:
#                 break

#             remaining = indices[1:]
#             overlaps = [calculate_iou(boxes[current], boxes[i]) for i in remaining]
#             distances = [calculate_center_distance(boxes[current], boxes[i]) for i in remaining]
            
#             # Apply both IoU threshold and distance threshold
#             indices = remaining[(np.array(overlaps) <= iou_threshold) & (np.array(distances) >= distance_threshold)]

#         final_predictions.extend(keep)

#     # Cross-Class NMS
#     final_boxes = np.array([pred['box'] for pred in final_predictions])
#     final_confidences = np.array([pred['conf'] for pred in final_predictions])
#     final_classes = np.array([pred['cls'] for pred in final_predictions])

#     indices = np.argsort(final_confidences)[::-1]  # Sort across classes by confidence
#     final_keep = []

#     while len(indices) > 0:
#         current = indices[0]
#         final_keep.append(final_predictions[current])

#         if len(indices) == 1:
#             break

#         remaining = indices[1:]
#         overlaps = [calculate_iou(final_boxes[current], final_boxes[i]) for i in remaining]
#         distances = [calculate_center_distance(final_boxes[current], final_boxes[i]) for i in remaining]

#         # Remove boxes that overlap or are too close across classes
#         indices = remaining[(np.array(overlaps) <= iou_threshold) | 
#                             (np.array(distances) >= distance_threshold) | 
#                             (final_classes[remaining] == final_classes[current])]

#     return final_keep

# # Run predictions on the test image
# results = model.predict(source=test_images_dir, save=False)

# # Handle prediction results
# for result in results:
#     image_path = result.path
#     img = cv2.imread(image_path)

#     # Prepare predictions
#     predictions = []
#     for box in result.boxes:
#         current_box = box.xyxy[0].cpu().numpy()
#         current_conf = box.conf.item()
#         current_cls = int(box.cls.item())

#         predictions.append({
#             "box": current_box,
#             "conf": current_conf,
#             "cls": current_cls
#         })

#     # Apply Non-Maximum Suppression (NMS) to resolve overlapping boxes
#     final_predictions = non_max_suppression(predictions, iou_threshold=0.1, distance_threshold=20)

#     for pred in final_predictions:
#         box = pred["box"]
#         conf = pred["conf"]
#         class_id = pred["cls"]
#         class_name = class_labels[class_id]  # Class name using the mapping

#         # Get the color and thickness for the class
#         color = class_color_thickness[class_name]['color']
#         thickness = class_color_thickness[class_name]['thickness']

#         # Draw the bounding box
#         cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness)

#         # Put the class label and confidence
#         label = f"{class_name}: {conf:.2f}"
#         cv2.putText(img, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

#     # Save the annotated image
#     output_image_path = os.path.join(output_dir, os.path.basename(image_path))
#     cv2.imwrite(output_image_path, img)

#     print(f"Annotated image saved to: {output_image_path}")

# # Output path for the annotated images
# print(f"All annotated images saved to: {output_dir}")




#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#evaluation 
#need to change the data.yaml on valid to test first

# from ultralytics import YOLO
# import os

# # Load the custom-trained model
# weights_path = r'C:\Internship Project YoloV11\ultralytics\runs\detect\toilet-bowl1\detect\train\weights\best.pt'
# model = YOLO(weights_path)

# # Path to the test dataset YAML file
# test_data_yaml_path = r'C:\Internship Project YoloV11\toilet-bowl\dataset.yml'  # Adjust the path accordingly

# # Output directory for results
# #output_dir = r'C:\Internship Project YoloV10\yolov10\runs\detect\train\results_test_metric'  # Specify your output directory

# if __name__ == '__main__':
#     # Run predictions on the test dataset
#     metrics = model.val(data=test_data_yaml_path, save_json=True, save_txt=True, save_conf=True)

#     # Save the results in the specified output directory
#     # if metrics:
#     #     with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
#   #         f.write(str(metrics))





#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

















# with IoU handling for four model 

# import os
# import cv2
# from ultralytics import YOLO
# import numpy as np

# # Define paths to the model weights
# weights_paths = [
#     r'C:\Internship Project YoloV11\ultralytics\runs\detect\exp37.1-nicks\weights\best.pt',
#     r'C:\Internship Project YoloV11\ultralytics\runs\detect\exp37.1-dents\weights\best.pt',
#     r'C:\Internship Project YoloV11\ultralytics\runs\detect\exp37.1-scratches\weights\best.pt',
#     r'C:\Internship Project YoloV11\ultralytics\runs\detect\exp37.1-pittings\weights\best.pt'
# ]

# test_images_dir = r'C:\Internship Project YoloV11\dataset_real_train_valid_test_intern\test\images'
# output_dir = r'C:\Internship Project YoloV11\ultralytics\runs\detect\exp37.1 four model results personal annotation with iou'

# # Create output directory if it does not exist
# os.makedirs(output_dir, exist_ok=True)

# # Load models
# models = [YOLO(weights_path) for weights_path in weights_paths]

# # Class-specific color codes and box thicknesses
# class_color_thickness = {
#     'Nicks': {'color': (60, 60, 255), 'thickness': 2},     # Nicks (Red) in BGR
#     'Dents': {'color': (148, 156, 255), 'thickness': 2},   # Dents (Light Red) in BGR
#     'Scratches': {'color': (28, 116, 255), 'thickness': 2}, # Scratches (Orange) in BGR
#     'Pittings': {'color': (28, 180, 255), 'thickness': 2}   # Pittings (Yellow) in BGR
# }

# # Mapping of class indices to class names
# class_labels = {0: 'Nicks', 1: 'Dents', 2: 'Scratches', 3: 'Pittings'}

# def calculate_iou(box1, box2):
#     x1 = max(box1[0], box2[0])
#     y1 = max(box1[1], box2[1])
#     x2 = min(box1[2], box2[2])
#     y2 = min(box1[3], box2[3])

#     inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
#     box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
#     box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

#     iou = inter_area / float(box1_area + box2_area - inter_area)
#     return iou

# def resolve_conflicts(predictions):
#     final_results = []
#     for pred in predictions:
#         for box in pred.boxes:
#             current_box = box.xyxy[0].cpu().numpy()
#             current_conf = box.conf[0].item()
#             current_cls = int(box.cls[0].item())

#             conflict_found = False
#             for final in final_results:
#                 final_box = final["box"]
#                 iou = calculate_iou(current_box, final_box)

#                 if iou > 0.3:
#                     conflict_found = True
#                     if current_conf > final["conf"]:
#                         final["box"] = current_box
#                         final["conf"] = current_conf
#                         final["cls"] = current_cls
#                     elif current_conf < 0.8:
#                         continue
#                     break
            
#             if not conflict_found:
#                 final_results.append({
#                     "box": current_box,
#                     "conf": current_conf,
#                     "cls": current_cls
#                 })
    
#     return final_results

# def combine_predictions(predictions_list):
#     combined_results = []
#     for predictions in predictions_list:
#         for result in predictions:
#             combined_results.append(result)
#     return combined_results

# def process_image_with_models(image_path, models):
#     all_predictions = []
#     image = cv2.imread(image_path)

#     for model in models:
#         print(f"Processing {image_path} with model {model}")
#         result = model.predict(source=image_path, save=False)
#         all_predictions.append(result)
        
#         # Optional: Save intermediate results for visualization
#         intermediate_result_path = os.path.join(output_dir, f"intermediate_{os.path.basename(image_path)}")
#         for r in result:
#             r.plot(intermediate_result_path)
        
#     combined_results = combine_predictions(all_predictions)
#     final_results = resolve_conflicts(combined_results)
    
#     return final_results

# # Process all images
# for image_name in os.listdir(test_images_dir):
#     image_path = os.path.join(test_images_dir, image_name)
#     combined_results = process_image_with_models(image_path, models)

#     img = cv2.imread(image_path)
#     for result in combined_results:
#         box = result["box"]
#         conf = result["conf"]
#         cls = result["cls"]

#         # Get the class name from the integer class
#         class_name = class_labels.get(cls)
#         if class_name is None:
#             print(f"Warning: Class {cls} not found in color mapping.")
#             continue

#         # Get the color and thickness for the class
#         color = class_color_thickness[class_name]['color']
#         thickness = class_color_thickness[class_name]['thickness']

#         # Draw the bounding box
#         cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        
#         # Put the class label and confidence
#         label = f"{class_name},{conf:.2f}"
       
#         cv2.putText(img, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     # Save the final image with annotations
#     final_result_path = os.path.join(output_dir, f"final_{image_name}")
#     cv2.imwrite(final_result_path, img)

#     print(f"Final annotated image saved to: {final_result_path}")
