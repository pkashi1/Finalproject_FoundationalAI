# Predicting the Assembly of Polymers in Surfactant/Water Medium with Machine Learning

## Project Overview
This project explores the prediction of polymer assembly in a surfactant/water medium using two regression models: **MLP Linear Regression** and **Transformer Regression**. The primary focus is on **ES Polymers**, composed of hydrophilic (E) and hydrophobic (S) monomer chains. The goal is to understand how the sequence of these monomers influences polymer structure and dynamics, specifically properties like **Radius of Gyration (RG)**, **Radial Distribution Function (RDF)**, and **Area (AREA)**.
We begin by analyzing correlations between the sequence of monomers and output properties, then use an MLP model to explore these relationships. Due to suboptimal accuracy with the MLP, we shift to a **Transformer Regression model**, which better captures the complex dependencies within polymer sequences by leveraging positional and sequence embeddings.

## Methodology
1. **Data Preprocessing:** Input sequences are tokenized, converted into indices, and padded to a fixed length of 21 tokens. The target variables are scaled using **RobustScaler**.
2. **MLP Model:** A basic linear regression model is used to predict polymer properties based on sequence features.
3. **Transformer Model:** The transformer uses **positional encoding** and **sequence embeddings** to capture long-range dependencies. A dropout rate of 0.1 is applied to prevent overfitting.
4. **Model Training:** The model is trained using **Adam optimizer** with a learning rate of 1e-3 and Mean Squared Error (MSE) loss over 25 epochs. The dataset is split into training and validation sets (80%/20%).

## Evaluation
The model is evaluated based on **MSE**, **R²**, and **Adjusted R²** scores. Results are compared for each target property (RG, RDF, AREA), and the **Transformer model** provides improved accuracy, especially for complex relationships between monomers.

## Files Included
- **train.py**: Contains the implementation of the model, training loop, and evaluation.
- **analysis.ipynb**: Data preprocessing, feature engineering, and model evaluation steps are presented.

## Data is not included as it is confidential
