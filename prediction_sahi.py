# from sahi import AutoDetectionModel
# from sahi.utils.cv import read_image
# from sahi.utils.file import download_from_url
# from sahi.predict import get_prediction, get_sliced_prediction, predict
# from IPython.display import Image


# from sahi.utils.yolov8 import (
#     download_yolov8s_model, download_yolov8s_seg_model
# )

# # Download YOLO11 model
# yolov8_model_path = r"C:\Internship Project YoloV11\ultralytics\runs\detect\others\leak valid 7%\weights\best.pt"

# detection_model = AutoDetectionModel.from_pretrained(
#     model_type='yolov8',
#     model_path=yolov8_model_path,
#     confidence_threshold=0.3,
#     device="cpu", # or 'cuda:0'
# )

# result = get_sliced_prediction(
#     r"C:\Internship Project YoloV11\Defects-101.BMP",
#     detection_model,
#     slice_height = 256,
#     slice_width = 256,
#     overlap_height_ratio = 0.2,
#     overlap_width_ratio = 0.2,
#     postprocess_type='NMS',
#     postprocess_match_metric='IOU',
#     postprocess_match_threshold=0.1,
#     postprocess_class_agnostic=True
# )

# #result = get_sliced_prediction(r"C:\Internship Project YoloV11\image.BMP", detection_model)

# result.export_visuals(export_dir=r"C:\Internship Project YoloV11")

# Image(r"C:\Internship Project YoloV11\prediction_visual.png")






#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



#prediction looping 


# import os
# from sahi import AutoDetectionModel
# from sahi.utils.cv import read_image
# from sahi.predict import get_sliced_prediction

# # Path to YOLOv11 model
# #yolov8_model_path = r"C:\Internship Project YoloV11\ultralytics\runs\detect\before doing paper\train4\weights\best.pt"
# yolov8_model_path = r"C:\Users\jovian sanjaya putra\runs\detect\train\weights\best.pt"


# # # Directory containing test images
# # test_images_dir = r"C:\Internship Project YoloV11\dataset_real_train_valid_test_intern\test\images"

# # # Directory to save the prediction visuals
# # output_dir = r"C:\Users\jovian sanjaya putra\runs\detect\best_one_model"

# test_images_dir = r"C:\test\dents\ori"

# output_dir = r"C:\test\dents\one model"

# # Create output directory if it doesn't exist
# os.makedirs(output_dir, exist_ok=True)

# # Initialize the detection model
# detection_model = AutoDetectionModel.from_pretrained(
#     model_type='yolov8',
#     model_path=yolov8_model_path,
#     confidence_threshold=0.4,
#     device="cuda:0",  # or 'cuda:0'
# )

# # Loop through all images in the test images directory
# for image_filename in os.listdir(test_images_dir):
#     # Ensure we are only processing image files
#     if image_filename.endswith(('.png', '.jpg', '.jpeg', '.BMP')):
#         image_path = os.path.join(test_images_dir, image_filename)
        
#         # Perform sliced prediction
#         result = get_sliced_prediction(
#             image_path,
#             detection_model,
#             slice_height=256,
#             slice_width=256,
#             overlap_height_ratio=0.2,
#             overlap_width_ratio=0.2,
#             postprocess_type='NMS',
#             postprocess_match_metric='IOU',
#             postprocess_match_threshold=0.1,
#             postprocess_class_agnostic=True,
#             verbose=1
#         )
        
#         # Save the annotated image with the same name as the input image
#         annotated_image_path = os.path.join(output_dir, f"{image_filename}")
#         result.export_visuals(
#             export_dir=output_dir, 
#             file_name=f"{image_filename}",
#             text_size=0.4,
#             rect_th=2
#         )


#         # print(result.to_coco_predictions())
        
#         # Optionally, you can print progress
#         print(f"Processed and saved annotated result for {image_filename}")

# # Optionally, you can print the completion message
# print("All images have been processed and saved.")



#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



#Four Model Sahi 


import os
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2


# Paths to the YOLOv8 models
weights_paths = [
    r"C:\Users\jovian sanjaya putra\runs\detect\best_nicks\weights\best.pt",
    r"C:\Users\jovian sanjaya putra\runs\detect\pending\best_dents\weights\best.pt",   
    #r"C:\Internship Project YoloV11\ultralytics\runs\detect\before doing paper\train4\weights\best.pt",
    r"C:\Users\jovian sanjaya putra\runs\detect\pending\best_scratches\weights\best.pt",
    r"C:\Users\jovian sanjaya putra\runs\detect\best_pittings\weights\best.pt",
]




# Directory containing test images
#test_images_dir = r"C:\Internship Project YoloV11\dataset_real_train_valid_test_intern\test\images"


test_images_dir = r"C:\test\dents\ori"

# Directory to save the prediction visuals
#output_dir = r"C:\Users\jovian sanjaya putra\runs\detect\best_four_model_results"

output_dir = r"C:\test\dents\four model"



os.makedirs(output_dir, exist_ok=True)

# Define class labels and corresponding color-coding for visualization
class_color_thickness = {
    'Nicks': {'color': (60, 60, 255), 'thickness': 2},
    'Dents': {'color': (148, 156, 255), 'thickness': 2},
    'Scratches': {'color': (28, 116, 255), 'thickness': 2},
    'Pittings': {'color': (28, 180, 255), 'thickness': 2}
}
class_labels = {0: 'Nicks', 1: 'Dents', 2: 'Scratches', 3: 'Pittings'}

# Load detection models
models = [
    AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=path,
        confidence_threshold=0.5,
        device='cuda:0'
    )
    for path in weights_paths
]

def process_image_with_models(image_path, models):
    """
    Process an image using multiple models and resolve conflicts among predictions.
    """
    all_predictions = []

    for model in models:
        # Get predictions using SAHI's slicing
        result = get_sliced_prediction(
            image_path,
            model,
            slice_height=256,
            slice_width=256,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            postprocess_type='NMS',
            postprocess_match_metric='IOU',
            postprocess_match_threshold=0.1,
            postprocess_class_agnostic=True,
            verbose=2
        )
        all_predictions.extend(result.to_coco_annotations())

    return resolve_conflicts(all_predictions)

def resolve_conflicts(predictions):
    """
    Resolve conflicts in predictions based on IOU and confidence scores.
    """
    final_results = []
    for pred in predictions:
        current_bbox = pred['bbox']
        current_conf = pred['score']
        current_cls = pred['category_id']

        conflict_found = False
        for final in final_results:
            iou = calculate_iou(current_bbox, final['bbox'])
            if iou > 0.1:  # Threshold for overlapping boxes
                conflict_found = True
                if current_conf > final['score']:
                    final.update({'bbox': current_bbox, 'score': current_conf, 'category_id': current_cls})
                break

        if not conflict_found:
            final_results.append(pred)

    return final_results

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) for two bounding boxes.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    box1_area = w1 * h1
    box2_area = w2 * h2

    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0

    return iou

# Process all test images
for image_name in os.listdir(test_images_dir):
    image_path = os.path.join(test_images_dir, image_name)
    predictions = process_image_with_models(image_path, models)

    # Load the image for visualization
    img = cv2.imread(image_path)

    for pred in predictions:
        box = pred['bbox']
        cls = pred['category_id']
        conf = pred['score']
        class_name = class_labels[cls]

        # Draw bounding box and label
        color = class_color_thickness[class_name]['color']
        thickness = class_color_thickness[class_name]['thickness']
        x, y, w, h = map(int, box)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        label = f"{class_name}, {conf:.2f}"
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    # Save the final annotated image
    output_path = os.path.join(output_dir, f"annotated_{image_name}")
    cv2.imwrite(output_path, img)
    print(f"Saved: {output_path}")
