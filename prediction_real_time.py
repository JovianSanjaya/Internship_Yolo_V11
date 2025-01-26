# import cv2
# from ultralytics import YOLO

# weight_path = r"C:\Internship Project YoloV11\ultralytics\runs\detect\others\leak valid 7%\weights\best.pt"
# # Load YOLOv8 model
# model = YOLO(weight_path)  # Use "yolov8s.pt" or "yolov8m.pt" for larger models

# # Define colors and thicknesses for each class
# colors = {
#     'Nicks': {'color': (60, 60, 255), 'thickness': 2},     # Nicks (Red) in BGR
#     'Dents': {'color': (148, 156, 255), 'thickness': 2},   # Dents (Light Red) in BGR
#     'Scratches': {'color': (28, 116, 255), 'thickness': 2}, # Scratches (Orange) in BGR
#     'Pittings': {'color': (28, 180, 255), 'thickness': 2}   # Pittings (Yellow) in BGR
# }

# # Initialize the camera
# cap = cv2.VideoCapture(0)  # Use 0 for the default camera

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Perform inference
#     results = model(frame)

#     # Extract the detections
#     detections = results[0].boxes

#     # Loop through the detections
#     for box in detections:
#         x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Bounding box coordinates
#         confidence = box.conf[0].cpu().numpy()      # Confidence score
#         class_id = int(box.cls[0].cpu().numpy())   # Class ID

#         # Get the class name
#         class_name = model.names[class_id]

#         # Get the color and thickness for the current class
#         if class_name in colors:
#             color = colors[class_name]['color']
#             thickness = colors[class_name]['thickness']
#         else:
#             color = (255, 0, 0)  # Default color (Blue) for unknown classes
#             thickness = 2

#         # Draw bounding box and label
#         label = f"{class_name}: {confidence:.2f}"
#         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
#         cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

#     # Display the resulting frame
#     cv2.imshow("YOLOv8 Real-Time Detection", frame)

#     # Break the loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the camera and close windows
# cap.release()
# cv2.destroyAllWindows()



# #-----------------------------------------------------------------------------------------------------




#with average rgb, confidence threhold, nsm threshold and detect every 50 frame

# import cv2
# from ultralytics import YOLO

# # Path to the YOLOv8 model weights
# weight_path = r"C:\Internship Project YoloV11\ultralytics\runs\detect\others\leak valid 7%\weights\best.pt"
# model = YOLO(weight_path)

# # Define colors and thicknesses for each class
# colors = {
#     'Nicks': {'color': (60, 60, 255), 'thickness': 2},
#     'Dents': {'color': (148, 156, 255), 'thickness': 2},
#     'Scratches': {'color': (28, 116, 255), 'thickness': 2},
#     'Pittings': {'color': (28, 180, 255), 'thickness': 2}
# }

# # Confidence and NMS IoU thresholds
# confidence_threshold = 0.55
# nms_iou_threshold = 0.3
# frame_check_interval = 60  # Perform inference every 60 frames
# frame_count = 0

# # Initialize the camera
# cap = cv2.VideoCapture(0)
# detections = []  # Store detections that pass the confidence check

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Only perform inference every 'frame_check_interval' frames
#     if frame_count % frame_check_interval == 0:
#         # Perform inference with confidence and IoU thresholds
#         results = model.predict(frame, conf=confidence_threshold, iou=nms_iou_threshold)
        
#         # Reset detections for each interval
#         detections = []
        
#         # Process each detected box
#         for box in results[0].boxes:
#             confidence = box.conf[0].cpu().numpy()
#             class_id = int(box.cls[0].cpu().numpy())
#             class_name = model.names[class_id]

#             # Check confidence against threshold
#             if confidence >= confidence_threshold:
#                 detections.append(box)

#     # Draw detections
#     for box in detections:
#         x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
#         confidence = box.conf[0].cpu().numpy()
#         class_id = int(box.cls[0].cpu().numpy())
#         class_name = model.names[class_id]

#         # Set color and thickness
#         color = colors.get(class_name, {'color': (255, 0, 0), 'thickness': 2})['color']
#         thickness = colors.get(class_name, {'color': (255, 0, 0), 'thickness': 2})['thickness']

#         # Draw bounding box and label
#         label = f"{class_name}: {confidence:.2f}"
#         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
#         cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

#     # Calculate the average RGB value of the entire frame
#     avg_rgb = frame.mean(axis=(0, 1))  # Mean along width and height (for each channel)

#     # Convert average RGB value to a string
#     avg_rgb_str = f"Avg RGB: ({int(avg_rgb[0])}, {int(avg_rgb[1])}, {int(avg_rgb[2])})"

#     # Display the average RGB value on the frame (e.g., top-left corner)
#     cv2.putText(frame, avg_rgb_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

#     # Display the resulting frame
#     cv2.imshow("YOLOv8 Real-Time Detection with Confidence Filter", frame)

#     # Increment frame count
#     frame_count += 1

#     # Break loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the camera and close windows
# cap.release()
# cv2.destroyAllWindows()





