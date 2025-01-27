
# Internship YoloV11 Documentation SIMTech

## Steps to Train the Model

### 1. Create and Activate Virtual Environment
- Create a virtual environment:
  ```bash
  python -m venv yolo_env
  ```
- Activate the virtual environment:
  - On Windows:
    ```bash
    yolo_env\Scripts\activate
    ```
  - On macOS/Linux:
    ```bash
    source yolo_env/bin/activate
    ```

### 2. Install Dependencies
 - Install the required libraries using:
    ```bash
    pip install -r requirements.txt
    ```

### 3. Configure `data.yaml`
- Create a `data.yaml` file in the dataset folder.
- The file should include paths to the training and validation datasets, as well as class labels. Example:
  ```yaml
  train: ./Internship Project YoloV11/dataset_real_train_valid_test_intern/train/images
  val: ./Internship Project YoloV11/dataset_real_train_valid_test_intern/valid/images
  test: ./Internship Project YoloV11/dataset_real_train_valid_test_intern/test/images
  
  
  
  nc: 4  # Number of classes
  names: ['Nicks', 'Dents', 'Scratches', 'Pittings']
  ```

### 4. Update `train.py`
- Update the `data` variable in `train.py` to point to your `data.yaml` file according to your dataset folder. Example:
  ```python
  data = './dataset/data.yaml'
  ```

### 5. Start Training
Run the following command in the virtual environment:
```bash
python train.py
```


&nbsp;

&nbsp;



## Modifications to Loss Function

### File Path: `ultralytics\ultralytics\utils\loss.py`

1. Under the `V8DetectionLoss` class, add the following line to define weights for each class:
   ```python
   weight = torch.FloatTensor([0.4148, 2.7776, 1.424, 1.8979]).to('cuda')
   ```

2. Modify the Binary Cross Entropy loss function:
   ```python
   self.bce = nn.BCEWithLogitsLoss(reduction="none", weight=weight)
   ```


**Note:** The number of weights should match the number of classes.


&nbsp;

&nbsp;



## BCEWithLogitsLoss Equation
The equation for Binary Cross Entropy with logits loss:

![image](https://github.com/user-attachments/assets/46145299-413e-4460-83eb-365b1a046d32)

## How to Calculate Weights
Use the formula:

![image](https://github.com/user-attachments/assets/b015e9af-c6ba-43ef-9032-a0c2ecba1740)

The implementation of this calculation is included in `calculate_weight.py`.



&nbsp;

&nbsp;



## Change Bounding Box Colors in Predictions

### File Path: `ultralytics/utils/plotting.py`

![image](https://github.com/user-attachments/assets/5cd5c2a6-d70f-4365-ab75-d2b5e293ae6c)


1. Navigate to the `Colors` class in `plotting.py`.
2. Modify the hex color codes for the first four classes:
   ```python
   Nicks: "FF3838",
   Dents: "FF9D97",
   Scratches: "FF701F",
   Pittings: "FFB21D"
   ```
   
## Additional Notes

### Example Directory Structure
```plaintext
./dataset
|-- train
|   |-- images
|   |-- labels
|-- valid
|   |-- images
|   |-- labels
|-- test
|   |-- images
|   |-- labels
|-- data.yaml
```

