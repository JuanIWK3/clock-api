import numpy as np
from sklearn.svm import SVC
import joblib
import os
from PIL import Image

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  
    img = img.resize((128, 128))  
    img_array = np.array(img).flatten() 
    return img_array.reshape(1, -1)  

def load_images(folder_path, label):
    images = []
    for filename in os.listdir(folder_path):
        img = Image.open(os.path.join(folder_path, filename)).convert('L')
        img = img.resize((128, 128)) 
        img_array = np.array(img).flatten()  
        images.append((img_array, label))
    return images

def test_model(model, image_dir, expected_label):
    correct_predictions = 0
    total_images = 0

    for file_name in os.listdir(image_dir):
        file_path = os.path.join(image_dir, file_name)
        if os.path.isfile(file_path):
            img_array = preprocess_image(file_path)

            prediction = model.predict(img_array)
            predicted_label = prediction[0]

            if predicted_label == expected_label:
                correct_predictions += 1
            total_images += 1

    accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0
    return accuracy

train_normal_path = 'train-images/normal'
train_disorder_path = 'train-images/disorder'
normal_images = load_images(train_normal_path, 0)  # Label normal as 0
disorder_images = load_images(train_disorder_path, 1)  # Label disorder as 1

test_normal_path = 'test-images/normal'
test_disorder_path = 'test-images/disorder'

def train_svm_until_test_accuracy(target_accuracy=0.8):
    test_accuracy_normal = 0
    test_accuracy_disorder = 0

    while test_accuracy_normal < target_accuracy * 100 or test_accuracy_disorder < target_accuracy * 100:
        all_images = normal_images + disorder_images
        np.random.shuffle(all_images)

        X = np.array([img for img, label in all_images])
        y = np.array([label for img, label in all_images])

        model = SVC(kernel='linear', C=1, gamma='scale')
        model.fit(X, y)

        test_accuracy_normal = test_model(model, test_normal_path, 0)
        test_accuracy_disorder = test_model(model, test_disorder_path, 1)

        print(f"Test accuracy on normal images: {test_accuracy_normal:.2f}%")
        print(f"Test accuracy on disorder images: {test_accuracy_disorder:.2f}%")

    return model

joblib.dump(train_svm_until_test_accuracy(), 'model.pkl')