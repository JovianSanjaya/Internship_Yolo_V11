
#To find the samllest and largest area


import os
import json
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Path to YOLOv8 model
yolov8_model_path = r"C:\Internship Project YoloV11 Web\object_detection\train_model.pt"

# Directory containing test images
test_images_dir = r"C:\temp\source_train"

# Directory to save the prediction visuals and JSON results
output_dir = r"C:\temp\result"
json_output_dir = r"C:\temp\result_json"

# Create directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(json_output_dir, exist_ok=True)

# Initialize the detection model
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=yolov8_model_path,
    confidence_threshold=0.4,
    device="cuda:0",
)

smallest_area = float('inf')
largest_area = 0

# Loop through all images in the test images directory
for image_filename in os.listdir(test_images_dir):
    if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        image_path = os.path.join(test_images_dir, image_filename)
        
        # Perform sliced prediction
        result = get_sliced_prediction(
            image_path,
            detection_model,
            slice_height=256,
            slice_width=256,
            overlap_height_ratio=0.2,                                   
            overlap_width_ratio=0.2,
            postprocess_type='NMS',
            postprocess_match_metric='IOU',
            postprocess_match_threshold=0.1,
            postprocess_class_agnostic=True,
            verbose=1
        )
        
        # Save the annotated image
        result.export_visuals(
            export_dir=output_dir, 
            file_name=image_filename,
            text_size=0.4,
            rect_th=2
        )
        
        # Extract COCO predictions
        coco_predictions = result.to_coco_predictions()
        
        # Store predictions in a JSON file
        json_filename = os.path.join(json_output_dir, image_filename.replace(" ", "_") + ".json")
        with open(json_filename, "w") as json_file:
            json.dump(coco_predictions, json_file, indent=4)

        # Find smallest and largest bounding box areas
        for obj in coco_predictions:
            area = obj["area"]
            smallest_area = min(smallest_area, area)
            largest_area = max(largest_area, area)

# Print smallest and largest area
print(f"Smallest bounding box area: {smallest_area}")
print(f"Largest bounding box area: {largest_area}")


#finding second largest area

# import os
# import json

# # Directory containing JSON results
# json_output_dir = r"C:\temp\result_json"

# areas = []

# # Loop through all JSON files in the result directory
# for json_filename in os.listdir(json_output_dir):
#     if json_filename.endswith(".json"):
#         json_path = os.path.join(json_output_dir, json_filename)
        
#         # Load JSON file
#         with open(json_path, "r") as json_file:
#             coco_predictions = json.load(json_file)
        
#         # Extract bounding box areas
#         for obj in coco_predictions:
#             areas.append(obj["area"])

# # Find the second-largest bounding box area
# if len(areas) < 2:
#     print("Not enough bounding boxes to determine the second-largest area.")
# else:
#     unique_areas = sorted(set(areas), reverse=True)  # Sort in descending order and remove duplicates
#     second_largest_area = unique_areas[1] + sec  if len(unique_areas) > 1 else unique_areas[0] 
#     print(f"Second-largest bounding box area: {second_largest_area}")


