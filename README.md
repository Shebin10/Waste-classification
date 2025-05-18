## Waste Classification using Deep Learning (MobileNetV2)

This project focuses on classifying different types of waste (like plastic, metal, paper, etc.) using deep learning. The goal is to support smarter waste management by automatically detecting waste types from images.

### Project Overview

- **Type:** Image Classification
- **Model Used:** MobileNetV2 (Transfer Learning)
- **Frameworks/Libraries:** TensorFlow, Keras, OpenCV, NumPy, Matplotlib
- **Dataset:** Custom image dataset with different waste categories

---

### Project Structure
waste_classification/
│
├── waste_venv/ # Virtual environment (ignored in Git)
├── dataset/ # Folder containing training and validation images
├── model/ # Trained model file (.h5 - not pushed to GitHub)
├── notebook/ # Jupyter notebooks (EDA, training, etc.)
├── requirements.txt # Python dependencies
├── .gitignore # Files and folders to ignore in Git
├── README.md # Project documentation
└── main.py / train.py # Main training script

### Key Steps in the Project

#### 1. **Data Preprocessing**
- Resized all images to a standard shape
- Performed image augmentation
- Split the dataset into training and validation sets

#### 2. **Model Building**
- Used **MobileNetV2** as the base model (pre-trained on ImageNet)
- Added custom classification layers on top
- Compiled the model using `Adam` optimizer and `categorical_crossentropy` loss

#### 3. **Training**
- Trained the model on the dataset
- Used early stopping and model checkpointing
- Achieved good accuracy and low validation loss

#### 4. **Evaluation**
- Plotted training and validation accuracy/loss graphs
- Evaluated the model on unseen images
- Displayed predictions with OpenCV
- Built a simple Streamlit app for real-time prediction and user-friendly interface

