# Predicting Kidney Disease: Automation Approach
This project focuses on predicting kidney disease using a machine learning approach, emphasizing automation in the model development and evaluation process. The core of this project is a Jupyter Notebook that demonstrates data preprocessing, model training, hyperparameter tuning, and comprehensive model evaluation.

## Features
The notebook, includes the following key features:
* Data Loading and Initial Exploration: Loads a cleaned dataset and displays its head and column names, indicating initial data preparation.
* Data Splitting: Divides the dataset into training and testing sets for model development and evaluation.
* Automated Machine Learning (PyCaret): Utilizes the pycaret.classification library for an automated approach to:
* Environment Setup: Configures the PyCaret environment for classification tasks, including data preprocessing (imputation for numeric and categorical features).
* Model Comparison: Automatically compares various classification models based on standard metrics like Accuracy, AUC, Recall, Precision, F1-Score, Kappa, and MCC.
* Model Creation: Selects and creates a specific model (e.g., Extra Trees Classifier, as seen in the output).
* Hyperparameter Tuning: Fine-tunes the hyperparameters of the selected model to improve performance.
* Model Evaluation Plots: Generates a variety of plots to assess model performance comprehensively, including:
> Pipeline Plot
> Hyperparameters
> AUC (Area Under the Curve)
> Confusion Matrix
> Threshold
> Precision Recall
> Prediction Error
> Class Report
> Feature Selection
> Learning Curve
> Manifold Learning
> Calibration Curve
> Validation Curve
> Dimensions
> Feature Importance
> Feature Importance (All)
> Decision Boundary
> Lift Chart
> Gain Chart
> Decision Tree
> KS Statistic Plot

* Prediction on Unseen Data: Demonstrates how to use the trained model to make predictions on new, unseen data.
* Model Persistence: Saves the trained machine learning pipeline and model to a .pkl file for future use.

## Technologies Used
* Python 3: The primary programming language used for the project.
* Jupyter Notebook / Google Colab: The interactive environment where the code is developed and executed.
* Pandas: For data manipulation and analysis.
* NumPy: For numerical operations.
* PyCaret: An open-source, low-code machine learning library in Python that automates machine learning workflows.
* Scikit-learn: Underlying machine learning library used by PyCaret.
* Matplotlib & Seaborn: For data visualization (though PyCaret handles plot generation internally).
* GPU Acceleration: The notebook metadata indicates the use of a T4 GPU, suggesting potential for faster model training.

### Getting Started
> To get a copy of the project up and running on your local machine for development and testing purposes, follow these steps.
> Prerequisites
    Python 3.x
     A Jupyter environment (e.g., Jupyter Notebook, JupyterLab, Google Colab).
  
