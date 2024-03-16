import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import requests
import io
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score

# Load the dataset of neutrino images
def load_images():
    data_larImages=requests.get('https://www.hep.ucl.ac.uk/undergrad/0056/other/projects/supernova/larImages.npy') #Load the l
    data_larImages.raise_for_status()
    images = np.load(io.BytesIO(data_larImages.content))/255 # Normalize pixel values to [0, 1]
    return images

# Simulate electronic noise in the detector
def add_electronic_noise(images, scale):
    # Initialize an array to store the noisy images
    noisy_images = np.empty_like(images)
    # Iterate through each image
    for i in range(len(images)):
        # Generate electronic noise with a normal distribution
        max_hit_values = np.max(images[i])
        noise = np.random.normal(loc=max_hit_values/scale, scale=max_hit_values/scale*10, size=images[i].shape)
        noisy_images[i] = images[i] + noise
        noisy_images[i] = np.clip(noisy_images[i], 0, 1)
    return noisy_images

# Prepare data for training and testing
def prepare_data(images, noise_levels):
    """
    Prepare data for training and testing a machine learning classifier.
    
    Args:
    - images (numpy.ndarray): Array of clean images.
    - noise_levels (list): List of noise levels for simulating electronic noise.
    
    Returns:
    - train_images (numpy.ndarray): Training images (clean and noisy).
    - test_images (numpy.ndarray): Testing images (clean and noisy).
    - train_labels (numpy.ndarray): Training labels (1 for clean, 0 for noisy).
    - test_labels (numpy.ndarray): Testing labels (1 for clean, 0 for noisy).
    """
    # Calculate noise scale based on noise levels
    
    # Generate noisy images for each noise level
    noisy_image_sets = {}  # Dictionary to store sets of noisy images

    for noise_level in noise_levels:
        noisy_images = add_electronic_noise(images, noise_level)
    noisy_image_sets[noise_level] = noisy_images

    # Concatenate clean images and noisy images for each noise level
    noisy_image_list = [noisy_image_sets[noise_level] for noise_level in noise_levels]
    all_images = np.concatenate([images] + noisy_image_list)
        
    

    
    # Generate labels for clean and noisy images
    clean_labels = np.ones(len(images))
    noisy_labels = np.zeros(sum([len(noise) for noise in noisy_images]))
    all_labels = np.concatenate([clean_labels, noisy_labels])
    
    # Split data into training and testing sets
    train_images, test_images, train_labels, test_labels = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)
    
    return train_images, test_images, train_labels, test_labels

# Train a CNN classifier
def train_classifier(train_images, train_labels):
    model = keras.Sequential([
        keras.layers.Conv2D(16, (3, 3), input_shape=(100, 100, 1)),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.Flatten(name= "feature_vectors"), 
        keras.layers.Dense(265),
        keras.layers.Activation('relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', 
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['accuracy'])
    # Train the model
    model.fit(train_images, train_labels, epochs=10, validation_split=0.2)
    
    return model

# Evaluate the classifier
def evaluate_classifier(model, test_images, test_labels):
    _, test_acc = model.evaluate(test_images, test_labels)
    print(f'Test accuracy: {test_acc}')

    threshold = 0.5
    label_pred = (model.predict(test_images) >= threshold).astype(int)
    cm = confusion_matrix(test_labels, label_pred)
    print(f'Confusion matrix:\n{cm}')

def evaluate_against_noise(images, noise_range, model):
    test_accuracies = []
    specificity_values = []
    recall_values = []

    for value in noise_range:
        noisy_images = add_electronic_noise(images, value)
        all_images = np.concatenate([images] + noisy_images)

        # Generate labels for clean and noisy images
        clean_labels = np.ones(len(images))
        noisy_labels = np.zeros(sum([len(noise) for noise in noisy_images]))
        all_labels = np.concatenate([clean_labels, noisy_labels])

        # Split data into training and testing sets
        train_images, test_images, train_labels, test_labels = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

        _, test_acc = model.evaluate(test_images, test_labels, verbose=0)
        test_accuracies.append(test_acc)

        y_pred = model.predict(test_images).round()
        tn, fp, fn, tp = confusion_matrix(test_labels, y_pred).ravel()
        specificity = tn / (tn + fp)
        recall = recall_score(test_labels, y_pred)
        specificity_values.append(specificity)
        recall_values.append(recall)

    plt.figure(figsize=(12, 6))
    plt.plot(noise_range, test_accuracies, label='Accuracy')
    plt.plot(noise_range, specificity_values, label='Specificity')
    plt.plot(noise_range, recall_values, label='Recall')
    plt.xlabel('Noise Level')
    plt.ylabel('Metrics')
    plt.title('Classifier Performance Against Noise Levels')
    plt.legend()
    plt.savefig('performance_against_noise.png')
    plt.show()

def save_image_pairs(images, noise_levels):
    clean_image = images[0]
    plt.imshow(clean_image, cmap='gray')
    plt.title('Clean Neutrino Image')
    plt.savefig(f'clean_neutrino.png')
    plt.close()
    for level in noise_levels:
        noisy_image = add_electronic_noise(clean_image, level)
        plt.imshow(noisy_image, cmap='gray')
        plt.title(f'Noisy Neutrino Image (Noise Level: {level})')
        plt.savefig(f'noisy_neutrino_noise_{level}.png')
        plt.close()

def main():
    # Load images
    images = load_images()
    
    # Set noise levels
    noise_levels = [0.33, 5.0, 20.0]  # Adjust as needed
    
    # Prepare data
    train_images, test_images, train_labels, test_labels = prepare_data(images, noise_levels)
    
    # Train classifier
    classifier = train_classifier(train_images, train_labels)
    
    # Evaluate classifier
    evaluate_classifier(classifier, test_images, test_labels)
    
    # Evaluate classifier against noise
    noise_range = np.linspace(0.0013725490196078432, 1.2470588235294118, 100)
    evaluate_against_noise(images, noise_range, classifier)

    # Save image pairs
    save_image_pairs(images, noise_levels)

if __name__ == "__main__":
    main()
