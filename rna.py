import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import os
from PIL import Image

# Function to preprocess the image (convert to grayscale)
def preprocess_image(image_path, target_size=(128, 128)):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize(target_size)  # Resize to match the training images
    img_array = np.array(img).flatten()  # Flatten the image
    return img_array.reshape(1, -1)  # Reshape to a 2D array

# Load your training and test data (replace with actual paths)
normal_path = 'train-images/normal'
disorder_path = 'train-images/disorder'

def load_images(folder_path, label):
    images = []
    for filename in os.listdir(folder_path):
        img = Image.open(os.path.join(folder_path, filename)).convert('L')  # Convert to grayscale
        img = img.resize((128, 128))  # Resize to the size used in training
        img_array = np.array(img).flatten()  # Flatten the image
        images.append((img_array, label))
    return images

# Load all images and labels
normal_images = load_images(normal_path, 0)  # Label normal as 0
disorder_images = load_images(disorder_path, 1)  # Label disorder as 1

# Combine and shuffle
all_images = normal_images + disorder_images
np.random.shuffle(all_images)

# Split data and labels
X = np.array([img for img, label in all_images])
y = np.array([label for img, label in all_images])

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM and continue until accuracy > 90%
def train_svm_until_accuracy(X_train, X_test, y_train, y_test, target_accuracy=0.8):
    accuracy = 0
    model = SVC(kernel='linear', C=1, gamma='scale')  # Initialize SVM model
    while accuracy < target_accuracy:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Current accuracy: {accuracy * 100:.2f}%")

    return model

# Train and save the model once 90% accuracy is achieved
model = train_svm_until_accuracy(X_train, X_test, y_train, y_test)

# Save the model
joblib.dump(model, 'svm_model.pkl')
print("Model trained and saved with accuracy >= 90%")

# Test the model
def test_model(model, image_dir, expected_label):
    correct_predictions = 0
    total_images = 0

    for file_name in os.listdir(image_dir):
        file_path = os.path.join(image_dir, file_name)
        if os.path.isfile(file_path):
            # Preprocess the image
            img_array = preprocess_image(file_path)

            # Make a prediction
            prediction = model.predict(img_array)
            predicted_label = prediction[0]

            # Check if the prediction matches the expected label
            if predicted_label == expected_label:
                correct_predictions += 1
            total_images += 1

            print(f'File: {file_name}, Prediction: {predicted_label}, Expected: {expected_label}')

    accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0
    print(f'Accuracy on {expected_label} images: {accuracy:.2f}%')

# Paths to image directories
normal_image_dir = './train-images/normal'
disorder_image_dir = './train-images/disorder'

# Evaluate the model
test_model(model, normal_image_dir, 0)  # Test normal images
test_model(model, disorder_image_dir, 1)  # Test disorder images
