# 🧠 Malaria Diagnosis with Convolutional Neural Networks (CNNs) using TensorFlow

This project demonstrates the full pipeline of building and optimizing deep learning models to diagnose **malaria** from cell images. Implemented in TensorFlow using both the Sequential and Functional APIs, it covers data preprocessing, model training, evaluation, augmentation, and deployment-ready improvements.

---

## 📂 Contents Overview

### 📝 Task Understanding
- Understanding the malaria classification task
- Overview of the dataset (Parasitized vs. Uninfected)

### 🧼 Data Preparation & Processing
- Loading image data using `tf.data`
- Resizing, normalizing, and shuffling
- Label encoding

### 📊 Data Visualization
- Visual inspection of image samples
- Grid display of input and augmented images

### 🏗️ Building Convolutional Neural Networks
- Building CNNs using `tf.keras.Sequential`
- Understanding the role of convolution, pooling, and dense layers
- Binary classification using sigmoid output

### ⚙️ Loss and Optimization
- Binary cross-entropy loss
- Adam optimizer with learning rate scheduling
- Custom loss functions and metrics

### 🚀 Model Training and Evaluation
- Training loop with validation
- Accuracy, precision, recall, and confusion matrix
- ROC plots

### 🧪 Model Saving and Loading
- Save and load models to/from Google Drive
- Use of TensorFlow callbacks for checkpoints

---

## 🧪 Advanced Topics

### 📚 Advanced Model Architectures
- Functional API usage
- Model subclassing and custom layers

### 🧠 Advanced Training Concepts
- Custom loss functions and metrics
- Eager vs. Graph mode
- Custom training loops

### 📈 TensorBoard Integration
- Logging metrics and visualizations
- Viewing model graphs
- Hyperparameter tuning and profiling

---

## 🧬 Data Augmentation Techniques

- Augmentation with `tf.image` and `Keras.layers`
- **Mixup** and **CutMix** strategies
- Powerful augmentation using **Albumentations**

---

## 🔧 MLOps with Weights & Biases
- Experiment tracking
- Hyperparameter sweeps
- Visualization of training runs

---

## 🛠 Tech Stack
- Python, TensorFlow, NumPy, Matplotlib
- Albumentations
- TensorBoard
- Weights & Biases (WandB)


