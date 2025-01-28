#show weights only training part first method to calculate 

# import os
# import torch

# def count_defects_and_images(images_dir, labels_dir):
#     # Initialize counts for defects and number of images containing each defect
#     defect_counts = {'Nicks': 0, 'Dents': 0, 'Scratches': 0, 'Pittings': 0}
#     images_with_defects = {'Nicks': 0, 'Dents': 0, 'Scratches': 0, 'Pittings': 0}
#     images_without_labels = 0
    
#     total_images = len([f for f in os.listdir(images_dir) if f.endswith('.BMP')])
#     total_labels = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
    
#     label_files = set(f.replace('.txt', '.BMP') for f in os.listdir(labels_dir) if f.endswith('.txt'))
#     image_files = [f for f in os.listdir(images_dir) if f.endswith('.BMP')]
    
#     # Track which images have labels
#     images_with_labels = set()
    
#     for lbl_file in os.listdir(labels_dir):
#         if lbl_file.endswith('.txt'):
#             img_file = lbl_file.replace('.txt', '.BMP')
#             if img_file in image_files:
#                 images_with_labels.add(img_file)
    
#     images_without_labels = len(set(image_files) - images_with_labels)
    
#     for lbl_file in os.listdir(labels_dir):
#         if lbl_file.endswith('.txt'):
#             img_file = lbl_file.replace('.txt', '.BMP')
#             if img_file not in image_files:
#                 continue  # Skip if the corresponding image file doesn't exist
            
#             has_defects = {'Nicks': False, 'Dents': False, 'Scratches': False, 'Pittings': False}
            
#             with open(os.path.join(labels_dir, lbl_file), 'r') as file:
#                 for line in file:
#                     parts = line.strip().split()
#                     if len(parts) > 0:
#                         class_id = int(parts[0])
#                         if class_id == 0:
#                             defect_counts['Nicks'] += 1
#                             has_defects['Nicks'] = True
#                         elif class_id == 1:
#                             defect_counts['Dents'] += 1
#                             has_defects['Dents'] = True
#                         elif class_id == 2:
#                             defect_counts['Scratches'] += 1
#                             has_defects['Scratches'] = True
#                         elif class_id == 3:
#                             defect_counts['Pittings'] += 1
#                             has_defects['Pittings'] = True
            
#             # Increment the image count for each defect found in the image
#             for defect, found in has_defects.items():
#                 if found:
#                     images_with_defects[defect] += 1

#     return defect_counts, images_with_defects, total_images, total_labels, images_without_labels

# def calculate_weights(defect_counts):
#     counts = torch.tensor([defect_counts['Nicks'], defect_counts['Dents'], 
#                            defect_counts['Scratches'], defect_counts['Pittings']], dtype=torch.float)
    
#     weights = 1 / counts

#     normalized_weights = weights / torch.min(weights)
    
#     return normalized_weights

# def main():
#     # Directories for training set
#     images_dir = r'C:\Internship Project YoloV10\dataset_real_train_valid_test_intern\train\images'
#     labels_dir = r'C:\Internship Project YoloV10\dataset_real_train_valid_test_intern\train\labels'

#     # Get defect counts and calculate weights for the training dataset
#     defect_counts, images_with_defects, total_images, total_labels, images_without_labels = count_defects_and_images(images_dir, labels_dir)
    
#     print("\nTraining Dataset:")
#     print(f"  Total Images: {total_images}")
#     print(f"  Total Labels: {total_labels}")
#     print(f"  Images Without Labels: {images_without_labels}")
#     print("Defect Counts:")
#     for defect, count in defect_counts.items():
#         print(f"  {defect}: {count}")

#     # Calculate and print class weights
#     weights = calculate_weights(defect_counts)
#     print(f"Class Weights for Training Dataset: {weights}")

# if __name__ == "__main__":
#     main()



#-------------------------------------------------------------------------------------------------------------------------------------------------------------

#show weights only training part second method to calculate

import os
import torch

def count_defects_and_images(images_dir, labels_dir):
    # Initialize counts for defects and number of images containing each defect
    defect_counts = {'Nicks': 0, 'Dents': 0, 'Scratches': 0, 'Pittings': 0}
    images_with_defects = {'Nicks': 0, 'Dents': 0, 'Scratches': 0, 'Pittings': 0}
    images_without_labels = 0
    
    total_images = len([f for f in os.listdir(images_dir) if f.endswith('.BMP')])
    total_labels = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
    
    label_files = set(f.replace('.txt', '.BMP') for f in os.listdir(labels_dir) if f.endswith('.txt'))
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.BMP')]
    
    # Track which images have labels
    images_with_labels = set()
    
    for lbl_file in os.listdir(labels_dir):
        if lbl_file.endswith('.txt'):
            img_file = lbl_file.replace('.txt', '.BMP')
            if img_file in image_files:
                images_with_labels.add(img_file)
    
    images_without_labels = len(set(image_files) - images_with_labels)
    
    for lbl_file in os.listdir(labels_dir):
        if lbl_file.endswith('.txt'):
            img_file = lbl_file.replace('.txt', '.BMP')
            if img_file not in image_files:
                continue  # Skip if the corresponding image file doesn't exist
            
            has_defects = {'Nicks': False, 'Dents': False, 'Scratches': False, 'Pittings': False}
            
            with open(os.path.join(labels_dir, lbl_file), 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        class_id = int(parts[0])
                        if class_id == 0:
                            defect_counts['Nicks'] += 1
                            has_defects['Nicks'] = True
                        elif class_id == 1:
                            defect_counts['Dents'] += 1
                            has_defects['Dents'] = True
                        elif class_id == 2:
                            defect_counts['Scratches'] += 1
                            has_defects['Scratches'] = True
                        elif class_id == 3:
                            defect_counts['Pittings'] += 1
                            has_defects['Pittings'] = True
            
            # Increment the image count for each defect found in the image
            for defect, found in has_defects.items():
                if found:
                    images_with_defects[defect] += 1

    return defect_counts, images_with_defects, total_images, total_labels, images_without_labels

def calculate_weights(defect_counts):

    total_samples = sum(defect_counts.values())
    num_classes = len(defect_counts)

    weights = [total_samples / (defect_counts[defect] * num_classes) if defect_counts[defect] > 0 else 0
               for defect in ['Nicks', 'Dents', 'Scratches', 'Pittings']]
    
    formatted_weights = [round(weight, 4) for weight in weights]
    
    return formatted_weights

def main():
    # Directories for training set
    images_dir = r'C:\Internship Project YoloV11\dataset_real_train_valid_test_intern\train\images'
    labels_dir = r'C:\Internship Project YoloV11\dataset_real_train_valid_test_intern\train\labels'

    # Get defect counts and calculate weights for the training dataset
    defect_counts, images_with_defects, total_images, total_labels, images_without_labels = count_defects_and_images(images_dir, labels_dir)
    
    print("\nTraining Dataset:")
    print(f"  Total Images: {total_images}")
    print(f"  Total Labels: {total_labels}")
    print(f"  Images Without Labels: {images_without_labels}")
    print("Defect Counts:")
    for defect, count in defect_counts.items():
        print(f"  {defect}: {count}")

    # Calculate and print class weights rounded to 4 decimal places
    weights = calculate_weights(defect_counts)
    print(f"Class Weights for Training Dataset: {weights}")

if __name__ == "__main__":
    main()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------


#considering train and valid

# import os

# def count_defects_and_images(images_dir, labels_dir):
#     # Initialize counts for defects and number of images containing each defect
#     defect_counts = {'Nicks': 0, 'Dents': 0, 'Scratches': 0, 'Pittings': 0}
#     images_with_defects = {'Nicks': 0, 'Dents': 0, 'Scratches': 0, 'Pittings': 0}
#     images_without_labels = 0
    
#     total_images = len([f for f in os.listdir(images_dir) if f.endswith('.BMP')])
#     total_labels = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
    
#     image_files = [f for f in os.listdir(images_dir) if f.endswith('.BMP')]
#     images_with_labels = set()

#     for lbl_file in os.listdir(labels_dir):
#         if lbl_file.endswith('.txt'):
#             img_file = lbl_file.replace('.txt', '.BMP')
#             if img_file in image_files:
#                 images_with_labels.add(img_file)

#     images_without_labels = len(set(image_files) - images_with_labels)
    
#     for lbl_file in os.listdir(labels_dir):
#         if lbl_file.endswith('.txt'):
#             img_file = lbl_file.replace('.txt', '.BMP')
#             if img_file not in image_files:
#                 continue  # Skip if the corresponding image file doesn't exist
            
#             has_defects = {'Nicks': False, 'Dents': False, 'Scratches': False, 'Pittings': False}
            
#             with open(os.path.join(labels_dir, lbl_file), 'r') as file:
#                 for line in file:
#                     parts = line.strip().split()
#                     if len(parts) > 0:
#                         class_id = int(parts[0])
#                         if class_id == 0:
#                             defect_counts['Nicks'] += 1
#                             has_defects['Nicks'] = True
#                         elif class_id == 1:
#                             defect_counts['Dents'] += 1
#                             has_defects['Dents'] = True
#                         elif class_id == 2:
#                             defect_counts['Scratches'] += 1
#                             has_defects['Scratches'] = True
#                         elif class_id == 3:
#                             defect_counts['Pittings'] += 1
#                             has_defects['Pittings'] = True
            
#             # Increment the image count for each defect found in the image
#             for defect, found in has_defects.items():
#                 if found:
#                     images_with_defects[defect] += 1

#     return defect_counts, images_with_defects, total_images, total_labels, images_without_labels

# def calculate_weights(total_defect_counts):
#     # Sum the total number of defect instances (total samples across all classes)
#     total_samples = sum(total_defect_counts.values())
#     num_classes = len(total_defect_counts)

#     # Calculate weights using the formula: total_samples / (num_samples_in_class_i * num_classes)

#     weights = [total_samples / (total_defect_counts[defect] * num_classes) if total_defect_counts[defect] > 0 else 0
#                for defect in ['Nicks', 'Dents', 'Scratches', 'Pittings']]
    
#     # Format the weights to 4 decimal places
#     formatted_weights = [round(weight, 4) for weight in weights]
    
#     return formatted_weights

# def main():
#     # Directories for training set
#     train_images_dir = r'C:\Internship Project YoloV11\dataset_real_train_valid_test_intern\train\images'
#     train_labels_dir = r'C:\Internship Project YoloV11\dataset_real_train_valid_test_intern\train\labels'
    
#     # Directories for validation set
#     valid_images_dir = r'C:\Internship Project YoloV11\dataset_real_train_valid_test_intern\valid\images'
#     valid_labels_dir = r'C:\Internship Project YoloV11\dataset_real_train_valid_test_intern\valid\labels'

#     # Get defect counts for the training dataset
#     train_defect_counts, _, train_total_images, train_total_labels, train_images_without_labels = count_defects_and_images(train_images_dir, train_labels_dir)
    
#     # Get defect counts for the validation dataset
#     valid_defect_counts, _, valid_total_images, valid_total_labels, valid_images_without_labels = count_defects_and_images(valid_images_dir, valid_labels_dir)
    
#     # Combine training and validation counts
#     total_defect_counts = {defect: train_defect_counts[defect] + valid_defect_counts[defect] for defect in train_defect_counts}
    
#     # Calculate and print combined class weights
#     weights = calculate_weights(total_defect_counts)
    
#     # Print combined defect counts and weights
#     print("\nCombined Dataset (Training + Validation):")
#     print(f"  Total Images: {train_total_images + valid_total_images}")
#     print(f"  Total Labels: {train_total_labels + valid_total_labels}")
#     print(f"  Images Without Labels: {train_images_without_labels + valid_images_without_labels}")
#     print("Combined Defect Counts:")
#     for defect, count in total_defect_counts.items():
#         print(f"  {defect}: {count}")

#     print(f"Class Weights for Combined Dataset: {weights}")

# if __name__ == "__main__":
#     main()




#-------------------------------------------------------------------------------------------------------------------------------------------------------------




#show weigths for three of them 

# import os
# import torch

# def count_defects_and_images(images_dir, labels_dir):
#     # Initialize counts for defects and number of images containing each defect
#     defect_counts = {'Nicks': 0, 'Dents': 0, 'Scratches': 0, 'Pittings': 0}
#     images_with_defects = {'Nicks': 0, 'Dents': 0, 'Scratches': 0, 'Pittings': 0}
#     images_without_labels = 0
    
#     total_images = len([f for f in os.listdir(images_dir) if f.endswith('.BMP')])
#     total_labels = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
    
#     label_files = set(f.replace('.txt', '.BMP') for f in os.listdir(labels_dir) if f.endswith('.txt'))
#     image_files = [f for f in os.listdir(images_dir) if f.endswith('.BMP')]
    
#     # Track which images have labels
#     images_with_labels = set()
    
#     for lbl_file in os.listdir(labels_dir):
#         if lbl_file.endswith('.txt'):
#             img_file = lbl_file.replace('.txt', '.BMP')
#             if img_file in image_files:
#                 images_with_labels.add(img_file)
    
#     images_without_labels = len(set(image_files) - images_with_labels)
    
#     for lbl_file in os.listdir(labels_dir):
#         if lbl_file.endswith('.txt'):
#             img_file = lbl_file.replace('.txt', '.BMP')
#             if img_file not in image_files:
#                 continue  # Skip if the corresponding image file doesn't exist
            
#             has_defects = {'Nicks': False, 'Dents': False, 'Scratches': False, 'Pittings': False}
            
#             with open(os.path.join(labels_dir, lbl_file), 'r') as file:
#                 for line in file:
#                     parts = line.strip().split()
#                     if len(parts) > 0:
#                         class_id = int(parts[0])
#                         if class_id == 0:
#                             defect_counts['Nicks'] += 1
#                             has_defects['Nicks'] = True
#                         elif class_id == 1:
#                             defect_counts['Dents'] += 1
#                             has_defects['Dents'] = True
#                         elif class_id == 2:
#                             defect_counts['Scratches'] += 1
#                             has_defects['Scratches'] = True
#                         elif class_id == 3:
#                             defect_counts['Pittings'] += 1
#                             has_defects['Pittings'] = True
            
#             # Increment the image count for each defect found in the image
#             for defect, found in has_defects.items():
#                 if found:
#                     images_with_defects[defect] += 1

#     return defect_counts, images_with_defects, total_images, total_labels, images_without_labels

# def calculate_weights(defect_counts):
#     # Convert defect counts to a tensor
#     counts = torch.tensor([defect_counts['Nicks'], defect_counts['Dents'], 
#                            defect_counts['Scratches'], defect_counts['Pittings']], dtype=torch.float)
    
#     # Calculate inverse frequency weights
#     weights = 1 / counts
#     # Normalize weights by dividing by the minimum weight to keep relative ratios
#     normalized_weights = weights / torch.min(weights)
    
#     return normalized_weights

# def main():
#     # Directories for dataset splits
#     images_dirs = {
#         'train': r'C:\Internship Project YoloV10\dataset_real_train_valid_test_intern\train\images',
#         'valid': r'C:\Internship Project YoloV10\dataset_real_train_valid_test_intern\valid\images',
#         'test': r'C:\Internship Project YoloV10\dataset_real_train_valid_test_intern\test\images'
#     }

#     labels_dirs = {
#         'train': r'C:\Internship Project YoloV10\dataset_real_train_valid_test_intern\train\labels',
#         'valid': r'C:\Internship Project YoloV10\dataset_real_train_valid_test_intern\valid\labels',
#         'test': r'C:\Internship Project YoloV10\dataset_real_train_valid_test_intern\test\labels'
#     }

#     # Collect and print defect counts, images with defects, total images, total labels, and images without labels for each dataset split
#     for split in ['train', 'valid', 'test']:
#         images_dir = images_dirs[split]
#         labels_dir = labels_dirs[split]
#         defect_counts, images_with_defects, total_images, total_labels, images_without_labels = count_defects_and_images(images_dir, labels_dir)
        
#         print(f"\n{split.capitalize()} Dataset:")
#         print(f"  Total Images: {total_images}")
#         print(f"  Total Labels: {total_labels}")
#         print(f"  Images Without Labels: {images_without_labels}")
#         print("Defect Counts:")
#         for defect, count in defect_counts.items():
#             print(f"  {defect}: {count}")

#         # Calculate and print class weights
#         weights = calculate_weights(defect_counts)
#         print(f"Class Weights: {weights}")

# if __name__ == "__main__":
#     main()
