import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score
from electronic_noise import load_images, train_classifier, evaluate_classifier, evaluate_against_noise
import math


mean = [75, 25]  # mean of distribution
cov = [[10, 0], [0, 10]]  # covariance matrix
g = np.random.multivariate_normal(mean, cov, size=(20))  # Multivariate distribution function returns a small distribution of points wrt the 2-dimensions of the image
g.shape
# mean = pixel position
# cov = spread of pixels
# size = density of points

rounded_up_array_axis_0 = [math.ceil(i) for i in (g[:, 0])]
rounded_up_array_axis_1 = [math.ceil(i) for i in (g[:, 1])]

def radionoise(images , scale):
    if images is None:
        images = np.load('larImages.npy')
    
    for k in range(len(images)):
        max_hit_value = np.max(images[k])
        for i in rounded_up_array_axis_0:
            for j in rounded_up_array_axis_1:
                images[k, i, j] = images[k, i, j] + np.random.normal(loc=max_hit_value/scale, scale=max_hit_value/(scale*10), size=1)
        # imagesori[k] = imagesori[k] - np.min(imagesori[k])
        #images[k] = images[k] / np.max(images[k])
    return images



def evaluate_against_noise(image, labels, noise_range, model):
    test_accuracies = []
    specificity_values = []
    recall_values = []

    for value in noise_range:

        images = image[value]
        labels = labels[value]
        # Split data into training and testing sets

        _, test_acc = model.evaluate(images, labels, verbose=0)
        test_accuracies.append(test_acc)

        y_pred = model.predict(images).round()
        tn, fp, fn, tp = confusion_matrix(labels, y_pred).ravel()
        specificity = tn / (tn + fp)
        recall = recall_score(labels, y_pred)
        specificity_values.append(specificity)
        recall_values.append(recall)

    plt.figure(figsize=(12, 6))
    plt.plot(noise_range, test_accuracies, label='Accuracy')
    plt.plot(noise_range, specificity_values, label='Specificity')
    plt.plot(noise_range, recall_values, label='Recall')
    plt.xlabel('Noise Distribution-Scale Parameter')
    plt.ylabel('Metrics')
    plt.title('Classifier Performance Against Noise Levels')
    plt.legend()
    plt.savefig('performance_against_noise.png')
    plt.show()

def main():

    #scale_values = [0.75,1,1.5]

    #noisy_images = []
    #for value in (scale_values):
    #    noisy_image = radionoise(None, value)
    #    noisy_images.append(noisy_image)


    #fig, ax = plt.subplots(1, 3, figsize=(20, 20))
    #ax[0].imshow(noisy_images[0][1])
    #ax[0].set_title(f"Noisy Neutrino Image (Scale: {scale_values[0]})", fontsize=10)
    #ax[1].imshow(noisy_images[1][1])
    #ax[1].set_title(f"Noisy Neutrino Image (Scale: {scale_values[1]})", fontsize=10)
    #ax[2].set_title(f"Noisy Neutrino Image (Scale: {scale_values[2]})", fontsize=10)
    #ax[2].imshow(noisy_images[2][1])
    #fig.tight_layout()
    #plt.savefig(f'noisy_neutrino_images.png')
    #plt.show()
    #plt.close()
    # Load images

    # Set noise levels
    noise_range = np.linspace(1, 5, 5) # Adjust as needed

    # Prepare data
    all_images = {} 
    all_labels = {} # Dictionary to store sets of noisy images
    for noise_level in noise_range:
        images_noisy = radionoise(None, noise_level)
        bimages = radionoise(np.empty((len(images_noisy),100,100)), noise_level)
        images_all = np.concatenate((bimages, images_noisy))
        labels_all = np.concatenate((np.ones(len(bimages)), np.zeros(len(images_noisy))))
        all_images[noise_level] = images_all
        all_labels[noise_level] = labels_all

    all_images_list = np.concatenate([all_images[noise_level] for noise_level in noise_range])
    all_labels_list = np.concatenate([all_labels[noise_level] for noise_level in noise_range])
    
    train_images, test_images, train_labels, test_labels = train_test_split(all_images_list, all_labels_list, test_size=0.2, random_state=42)
    
    # Train classifier
    classifier = train_classifier(train_images, train_labels)
    
    # Evaluate classifier
    evaluate_classifier(classifier, test_images, test_labels)
    
    #
    evaluate_against_noise(all_images, all_labels, noise_range, classifier)

if __name__ == "__main__":
    main()
