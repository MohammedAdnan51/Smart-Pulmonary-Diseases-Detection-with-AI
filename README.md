# AI-Driven Detection of Lung Diseases

This project focuses on the **early detection of lung diseases** such as **pneumonia, tuberculosis (TB), COVID-19**, and others using **machine learning** and **medical imaging techniques**. Leveraging deep learning architectures like **VGG16** and interpretability techniques like **Grad-CAM**, the project aims to develop an automated and scalable system for lung disease classification.

## Table of Contents
1. [Introduction](#introduction)
2. [Proposed Solution](#proposed-solution)
3. [Technologies Used](#technologies-used)
4. [Implementation](#implementation)
5. [Results](#results)
6. [Conclusion](#conclusion)

## Introduction
Lung diseases are a major global health concern, causing millions of deaths worldwide each year. **Chest X-rays** are commonly used to diagnose these conditions, but manual interpretation can be time-consuming and prone to errors. The project utilizes **deep learning** to automate the diagnosis of lung diseases and improve diagnostic accuracy.

### Purpose of the System
The system aims to:
- Provide **faster diagnosis** and treatment planning.
- Improve **accuracy** and **consistency** of diagnosis.
- Reduce the workload of radiologists.
- **Scale** efficiently to handle large datasets.

## Proposed Solution
- **Transfer Learning**: Utilized the pre-trained **VGG16** model on ImageNet to extract generic features such as edges and shapes. The model was fine-tuned with lung disease data to focus on disease-specific features.
- **Grad-CAM**: Implemented **Gradient-weighted Class Activation Mapping (Grad-CAM)** for model interpretability. Grad-CAM helps visualize the areas in the X-ray image that contributed to the prediction.
- **Fine-Tuning**: Fine-tuned deeper layers of VGG16 to improve classification performance, allowing the model to specialize in detecting subtle disease-specific patterns.

### Key Advantages
- High diagnostic accuracy.
- Scalability to handle large datasets.
- Early detection and improved treatment planning.
- Consistency and objectivity, reducing the potential for human error.

## Technologies Used
- **Programming Language**: Python
- **Deep Learning Framework**: TensorFlow, Keras
- **Image Processing**: OpenCV, PIL
- **Model Architecture**: VGG16
- **Visualization**: Grad-CAM
- **Libraries**: NumPy, Matplotlib, Tkinter (for GUI)

## Implementation
The project involves the following stages:
1. **Data Preprocessing**:
   - Images resized to 224x224 pixels and normalized for consistency.
   - Applied data augmentation techniques like flipping and rotation.
2. **Model Training**:
   - **VGG16** is used as the base model for feature extraction.
   - Added custom fully connected layers and a **softmax** output layer to classify diseases into five categories.
   - Fine-tuned the model to specialize in lung disease classification.
3. **Evaluation**:
   - **Accuracy, precision, recall**, and **F1-score** were used to assess model performance.
   - **ROC curves** and **Grad-CAM** visualizations were generated to highlight the affected lung areas and model predictions.

## Results
The model successfully classified various lung diseases, achieving high accuracy in predicting conditions like:
- **Bacterial Pneumonia**
- **Viral Pneumonia**
- **COVID-19**
- **Tuberculosis**
- **Normal (No Disease)**

### GUI
A simple GUI interface was built using **Tkinter** to allow users to upload chest X-ray images and get a prediction on the type of lung disease present.

## Conclusion
The proposed system demonstrates a significant advancement in the automated diagnosis of lung diseases using deep learning and medical imaging. By integrating **VGG16** and **Grad-CAM**, the system provides an accurate and interpretable solution for healthcare professionals.

## Future Work
- Fine-tuning the model on a more diverse dataset to further improve accuracy.
- Exploring other deep learning architectures for enhanced performance.

