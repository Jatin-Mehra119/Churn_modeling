# Churn Modeling

This repository contains a Jupyter Notebook for building and evaluating a model to predict customer churn. Customer churn refers to the loss of clients or customers. The notebook details the steps involved in data preprocessing, exploratory data analysis (EDA), feature engineering, model building, and evaluation.

## Table of Contents

1.  [Installation](#installation)
2.  [Data Description](#data-description)
3.  [Notebook Overview](#notebook-overview)
4.  [Modeling Process](#modeling-process)
5.  [Results](#results)
6.  [Conclusion](#conclusion)
7.  [References](#references)

## Installation

To run the notebook locally, ensure you have the following dependencies installed:

-   Python 3.x
-   Jupyter Notebook
-   Pandas
-   NumPy
-   Scikit-learn
-   Matplotlib
-   Seaborn

You can install the required packages using the following command:

`pip install pandas numpy scikit-learn matplotlib seaborn` 

## Data Description

The dataset contains customer-related information. Features include:

-   **CustomerID**: Unique identifier for each customer.
-   **Geography**: Customer's geographical location.
-   **Gender**: Customer's gender.
-   **Age**: Customer's age.
-   **Tenure**: Number of years the customer has been with the company.
-   **Balance**: Account balance of the customer.
-   **ProductsNumber**: Number of products the customer has with the company.
-   **HasCrCard**: Whether the customer has a credit card.
-   **IsActiveMember**: Whether the customer is an active member.
-   **EstimatedSalary**: Estimated salary of the customer.
-   **Exited**: Whether the customer has churned (target variable).

## Notebook Overview

The notebook consists of the following sections:

1.  **Data Loading and Preprocessing**:
    -   Import necessary libraries.
    -   Load the dataset.
    -   Handle missing values and perform data cleaning.
2.  **Exploratory Data Analysis (EDA)**:
    -   Visualize the distribution of various features.
    -   Analyze the correlation between features.
    -   Explore the relationship between features and the target variable (churn).
3.  **Feature Engineering**:
    -   Encode categorical variables.
    -   Scale numerical features.
4.  **Model Building**:
    -   Split the data into training and testing sets.
    -   Train various machine learning models (e.g., Logistic Regression, Decision Trees, Random Forest, etc.).
    -   Evaluate the performance of the models using appropriate metrics.
5.  **Model Evaluation**:
    -   Compare model performance using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
    -   Select the best-performing model.

## Modeling Process

The modeling process includes:

-   **Data Splitting**: Dividing the data into training and testing sets to evaluate model performance.
-   **Model Training**: Training various machine learning algorithms to find the best model.
-   **Hyperparameter Tuning**: Using techniques like Random Search to optimize model parameters.
-   **Model Evaluation**: Assessing models using cross-validation and performance metrics.

## Results

### Logistic Regression

-   **Accuracy**: 0.80
-   **Precision (Positive Class)**: 0.58
-   **Recall (Positive Class)**: 0.18
-   **F1-Score (Positive Class)**: 0.28
-   **ROC-AUC**: 0.77

### Random Forest Classifier

-   **Accuracy**: 0.85
-   **Precision (Positive Class)**: 0.72
-   **Recall (Positive Class)**: 0.47
-   **F1-Score (Positive Class)**: 0.57
-   **ROC-AUC**: 0.85

### Gradient Boosting Classifier

-   **Accuracy**: 0.87
-   **Precision (Positive Class)**: 0.78
-   **Recall (Positive Class)**: 0.49
-   **F1-Score (Positive Class)**: 0.60
-   **ROC-AUC**: 0.87

### AdaBoost Classifier

-   **Accuracy**: 0.85
-   **Precision (Positive Class)**: 0.71
-   **Recall (Positive Class)**: 0.47
-   **F1-Score (Positive Class)**: 0.57
-   **ROC-AUC**: 0.85

### Support Vector Machine (SVM)

-   **Accuracy**: 0.85
-   **Precision (Positive Class)**: 0.79
-   **Recall (Positive Class)**: 0.42
-   **F1-Score (Positive Class)**: 0.55
-   **ROC-AUC**: 0.83

### K-Nearest Neighbors Classifier

-   **Accuracy**: 0.82
-   **Precision (Positive Class)**: 0.60
-   **Recall (Positive Class)**: 0.44
-   **F1-Score (Positive Class)**: 0.51
-   **ROC-AUC**: 0.78

### Extra Trees Classifier

-   **Accuracy**: 0.86
-   **Precision (Positive Class)**: 0.75
-   **Recall (Positive Class)**: 0.46
-   **F1-Score (Positive Class)**: 0.57
-   **ROC-AUC**: 0.85

### MLP Classifier (Neural Networks)

-   **Accuracy**: 0.86
-   **Precision (Positive Class)**: 0.78
-   **Recall (Positive Class)**: 0.45
-   **F1-Score (Positive Class)**: 0.58
-   **ROC-AUC**: 0.86

## Conclusion

The **Gradient Boosting Classifier** is the most promising model with 87% accuracy. Random Forest, AdaBoost, SVM, Extra Trees, and MLP (neural network) also perform well, scoring around 85% to 86% accuracy. Logistic Regression and K-Nearest Neighbors Classifier score around 80% to 82% accuracy.

## References

-   [Scikit-learn Documentation](https://scikit-learn.org/stable/supervised_learning.html)
-   [Seaborn Documentation](https://seaborn.pydata.org/tutorial.html)