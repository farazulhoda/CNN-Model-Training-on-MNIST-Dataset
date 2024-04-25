# CNN Model Training on MNIST Dataset

## Overview

This repository contains code for training a Convolutional Neural Network (CNN) model on the MNIST dataset using TensorFlow and Keras. The MNIST dataset consists of grayscale images of handwritten digits from 0 to 9, and the goal of the model is to accurately classify these digits.

## Code Explanation

The provided code performs the following steps:

1. **Data Loading and Preprocessing**:
   - Loads the MNIST dataset using TensorFlow's `mnist.load_data()` function.
   - Normalizes the pixel values of the images to the range [0, 1] by dividing by 255.0.

2. **Model Architecture Definition**:
   - Defines a CNN model architecture using TensorFlow's Keras API.
   - The model consists of convolutional layers, max-pooling layers, flatten layer, and dense layers with ReLU activation functions.

3. **Model Compilation and Training**:
   - Compiles the model with the Adam optimizer, sparse categorical cross-entropy loss function, and accuracy metric.
   - Trains the model on the training data for 5 epochs, validating it on the test data.

4. **Model Evaluation**:
   - Evaluates the trained model's performance on the test data and prints the test accuracy.

5. **Adversarial Examples Generation**:
   - Generates adversarial examples by adding a small perturbation to the test data.
   - Evaluates the model's performance on the adversarial examples.

6. **Model Visualization**:
   - Generates a visualization of the model's architecture using TensorFlow's `plot_model()` function.
   - Plots the training and validation accuracy and loss over epochs using Matplotlib.

## Usage

To run the code, ensure you have TensorFlow, NumPy, and Matplotlib installed. You can execute the provided Python script in your preferred environment. Make sure to adjust any paths or configurations as needed.

```bash
# Clone the repository
git clone https://github.com/farazulhoda/CNN-Model-Training-on-MNIST-Dataset.git

# Navigate to the project directory
cd CNN-Model-Training-on-MNIST-Dataset

# Install dependencies
pip install -r requirements.txt

# Run the script
python train_cnn_mnist.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The code in this repository is adapted from TensorFlow's official documentation and tutorials.
- The MNIST dataset is widely used in the machine learning community and is provided by TensorFlow and Keras for educational purposes.

For more information, please refer to the [official TensorFlow documentation](https://www.tensorflow.org/) and [Keras documentation](https://keras.io/).