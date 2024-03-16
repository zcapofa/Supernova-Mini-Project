# Supernova Neutrino Detection using Machine Learning

This repository contains code for investigating the feasibility of identifying supernova neutrino events in a liquid argon time-projection chamber using machine learning techniques. The project includes simulating electronic noise in the detector, developing a machine learning classifier to distinguish clean neutrinos from noisy ones, and evaluating the classifier's performance under different noise levels.

# Files Included:
larImages.npy: A numpy array containing 10,000 100x100 pixel images representing energy deposition in the liquid argon detector.

meta.npy: Meta information about the particles in the images, including details such as neutrino energy, initial and final state particles, their PDG codes, total energy, momentum components, etc.

electronic_noise.py: Python script containing functions to simulate electronic noise in the detector and train/evaluate machine learning classifiers.

generate_noisy_image.py: Python script for generating noisy images by overlaying simulated noise on clean neutrino images. This script currently utilizes radioactive noise simulation.

# Machine Learning Tasks:
Electronic Noise Simulation:
Implemented a method to simulate electronic noise in the detector, following a normal distribution.

Radioactive Noise Simulation:
Developed a method to generate noise in the images simulating radioactive noise in the form of randomly placed Gaussian 'blobs' with appropriate energy.

Classifier Development:
Developed a machine learning classifier to classify clean neutrinos from noisy slices with varying levels of electronic noise.

Classifier Evaluation:
Tested the classifier on simulated neutrinos overlaid with different levels of electronic and radioactive noise to assess its robustness.
