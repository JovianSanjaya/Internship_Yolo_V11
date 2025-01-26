from ultralytics import YOLO

# Load a model
model = YOLO("yolo11s.pt")

if __name__ == "__main__":
    # Train the model with the additional parameters
    model.train(
        data=r'C:\Internship Project YoloV11\dataset_real_train_valid_test_intern\data.yaml',
        epochs=100, 
        batch=8, 
        imgsz=640,
        lr0=0.0007, 
        lrf=0.00005, 
        optimizer="SGD",
        weight_decay=0.0005, 
        patience=10,
        cos_lr=False,
        save=True,
        cls=0.5,
        device=0
    )


