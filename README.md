# ai-hw-spring-2026-aa

# FGSM Attack on MNIST Recognition

## Objective

Evaluate the robustness of a trained neural network by applying the Fast Gradient Sign Method (FGSM) adversarial attack to the MNIST handwritten digit dataset.

## Dataset Summary

The MNIST dataset consists of 70,000 28x28 black-and-white images of handwritten digits extracted from two NIST databases. There are 60,000 images in the training dataset and 10,000 images in the test dataset. The dataset contains ten classes representing the digits 0 through 9.

## Dataset

- MNIST Handwritten Digits Dataset
- Training Set: 60,000 images
- Test Set: 10,000 images
- Image Size: 28 × 28 pixels
- Number of Classes: 10

## Model

A shallow Multi-Layer Perceptron (MLP) trained in Assignment 4.1 was used as the target model for the adversarial attack.

## Architecture

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

The Fast Gradient Sign Method (FGSM) is one of the most common adversarial attack techniques used to evaluate the robustness of neural networks.

Instead of changing the model itself, FGSM slightly modifies the input image using the gradient of the loss function. These modifications are often imperceptible to humans but can significantly affect the model's predictions.

### Attack Process

```text
Original Image
        ↓
Compute Gradient
        ↓
Generate Adversarial Image
        ↓
Run Prediction Again
        ↓
Compare Results
```

### Attack Parameters

- Attack Method: FGSM
- Attack Strength (ε): 0.15

## Evaluation Metrics

The following metrics were used to evaluate the effectiveness of the attack:

### Recognition Rate Before Attack

Measures the classification accuracy of the original model on normal test images.

### Recognition Rate After Attack

Measures the classification accuracy after FGSM perturbations have been applied.

### Attack Success Rate (ASR)

Measures how successfully the attack causes originally correct predictions to become incorrect.

```text
ASR = (Original Correct - Attacked Correct) / Original Correct
```

## Results

- Recognition Rate Before Attack: 97.38%
- Recognition Rate After FGSM Attack: 1.07%
- Attack Success Rate (ASR): 98.90%

The FGSM attack reduced the model accuracy from 97.38% to 1.07%, demonstrating that the trained neural network is highly vulnerable to adversarial perturbations.

## Repository Structure

```text
model.py   - Defines the MLP architecture
train.py   - Trains the model using the MNIST training set
test.py    - Evaluates the model and performs the FGSM attack
```

## Conclusion

This project evaluated the robustness of a neural network against adversarial attacks using FGSM. Although the original model achieved high accuracy on the MNIST dataset, the attack dramatically reduced its performance. The results demonstrate that neural networks can be highly sensitive to carefully crafted perturbations, highlighting the importance of adversarial robustness in machine learning systems.
