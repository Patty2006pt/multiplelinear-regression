# multiplelinear-Regression Project (Advertising & Sales Data)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning demonstration project built with Python, Scikit-Learn, and Seaborn to perform Multiple Linear Regression, analyzing and predicting product sales based on advertising budgets across multiple channels (TV, Radio, and Newspaper).

🚀 Features
📊 Loads and inspects CSV advertising datasets using pandas.   
📉 Calculates correlation matrices to analyze relationships between multiple advertising channels and sales performance.   
📈 Implements Multiple Linear Regression via scikit-learn's LinearRegression model.   
📊 Visualizes multi-variable relationships and trends using Seaborn pair plots (pairplot).   
📐 Evaluates model accuracy using Mean Absolute Error (MAE) and generates multi-feature predictions.   

🛠️ Tech StackLanguage: 
- Python 3.xLibraries:pandas (Data manipulation and structuring)   
- numpy (Numerical operations)   
- scikit-learn (Machine learning modeling and evaluation metrics)   
- matplotlib & seaborn (Data visualization and pair plots)

⚙️ Getting Started
Prerequisites
Make sure you have Python installed along with the required analytical packages:
pip install pandas numpy scikit-learn matplotlib seaborn

Installation & Execution
- Place your dataset file (Advertising2.csv) in the working directory.
- Copy the code into your local Jupyter Notebook, Google Colab, or save it as a Python script.Run the script:
  python multiple_regression_ads.py

📖 UsageHere is a core snippet demonstrating how to load the dataset, check correlations, fit a Multiple Linear Regression model, and make predictions:   

import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load dataset
df = pd.read_csv("Advertising2.csv")

# Check correlation matrix
print(df.corr())

# Define independent variables (x) and target variable (y)
x = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Initialize and fit the Multiple Linear Regression model
lrm = LinearRegression()
lrm.fit(x, y)

# Make predictions for new advertising budgets (e.g., TV=10, Radio=20, Newspaper=30)
predictions = lrm.predict([[10, 20, 30]])
print("Predicted Sales:", predictions[0])

# Evaluate model performance using Mean Absolute Error
all_predictions = lrm.predict(x)
mae = metrics.mean_absolute_error(y, all_predictions)
print("Mean Absolute Error:", mae)
