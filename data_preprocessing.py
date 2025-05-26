import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Define the base directory where images are stored
data_dir = r"E:\Lung Cancer\LungCANCER\test_images"

# Define class labels (Fixed typo here)
classes = {'Benign case', 'Malignant case', 'Normal case'}

# Initialize empty lists for images and labels
images = []
labels = []

image_size = 224

# Loop through each class directory
for class_name in classes:
    # Get the path to the current class directory
    class_dir = os.path.join(data_dir, class_name)

    # Check if the directory exists
    if not os.path.exists(class_dir):
        print(f"Directory not found: {class_dir}")
        continue

    # Loop through each image file in the class directory
    for filename in os.listdir(class_dir):
        # Get the full path to the image file
        image_path = os.path.join(class_dir, filename)

        # Check if the file is an image
        if image_path.lower().endswith(('png', 'jpg', 'jpeg')):
            try:
                # Load the image using OpenCV
                image = cv2.imread(image_path)  # load the image
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert it to RGB for 3 color channels
                image = cv2.resize(image, (image_size, image_size))  # resize to 224x224

                # Append the image and its corresponding label to the lists
                images.append(image)
                labels.append(class_name)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")

# Encode all the labels to numeric values
label_dict = {"Benign case": 0, "Malignant case": 1, "Normal case": 2}
encoded_labels = [label_dict[label] for label in labels]

# Shuffle the data by generating a permutation of indices
permutation = np.random.permutation(len(encoded_labels))

X = np.array(images)
X = X / 255.  # Normalize the pixel scale to 0-1 (0 = black, 1 = white)
Y = np.array(encoded_labels)

# Shuffle images and labels
X = X[permutation]
Y = Y[permutation]

# Optional: Split the dataset into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
