# ai-hw-spring-2026-aa

# FGSM Attack MNIST Recognition

## Objective
Evaluate the robustness of a trained neural network by applying the Fast Gradient Sign Method (FGSM) adversarial attack to the MNIST handwritten digit dataset.

## Dataset Summary
The MNIST dataset consists of 70,000 28x28 black-and-white images of handwritten digits extracted from two NIST databases. There are 60,000 images in the training dataset and 10,000 images in the validation dataset, one class per digit so a total of 10 classes, with 7,000 images (6,000 train images and 1,000 test images) per class. Half of the image were drawn by Census Bureau employees and the other half by high school students (this split is evenly distributed in the training and testing sets).

## Dataset
- MNIST Handwritten Digits Dataset
- Training Set: 60,000 images
- Test Set: 10,000 images
- Image Size: 28 × 28 pixels

## Model
A shallow Multi-Layer Perceptron (MLP).

# Architecture
```text
Input Image (28x28)
        ↓
Flatten (784)
        ↓
Linear Layer (784 → 128)
        ↓
ReLU Activation
        ↓
Linear Layer (128 → 10)
        ↓
Digit Prediction (0-9)
```

## FGSM Attack

The Fast Gradient Sign Method (FGSM) generates adversarial examples by adding a small perturbation to the input image based on the gradient of the loss function.

- Attack Method: FGSM
- Attack Strength (ε): 0.15

## Results
- Recognition Rate Before Attack: 97.38%
- Recognition Rate After FGSM Attack: 1.07%
- Attack Success Rate (ASR): 98.90%

## Repository Structure

```text
model.py   - Defines the MLP architecture
train.py   - Trains the model using the MNIST training set
test.py    - Evaluates the model using the MNIST test set
