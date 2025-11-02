# Tower Failure Prediction Using Neural Networks

This project contains two neural network models designed to **predict tower failure under different load conditions**, based on data derived from a Finite Element Method (FEM) analysis.  

---

## 🧠 Overview

The goal of this project is to classify whether a tower will fail under various loading scenarios. The models take in FEM-calculated features and predict the likelihood of structural failure.

Two different neural network architectures are implemented and compared:

1. **ResNet-based Classifier**  
2. **U-Net-based Failure Map Predictor**

---

## ⚙️ Models

### 1. ResNet Classifier
The **ResNet** model classifies whether the tower fails or remains stable under given load conditions.

**Key features:**
- Adjustable **number of layers**, **layer size**, and **activation function** (non-linearity)
- Optimized using the **Adam** optimizer  
- Includes a **learning rate (LR) scheduler** for training stability

---

### 2. U-Net Predictor
The **U-Net** model outputs the **failure probability at each load point**, enabling users to visualize which loads are most likely to cause structural failure.

**Key features:**
- Adjustable **number of feature maps** in each layer
- Optimized using the **Adam** optimizer  
- Includes a **learning rate (LR) scheduler**  
- Offers interpretability by identifying **specific load cases** that contribute to failure

## 🙏 Acknowledgments

Special thanks to Dr. Burigede Liu and Ms. Rui Wu for providing the initial code framework as part of Course 4C11 at CUED.
