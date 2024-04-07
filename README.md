# Turkey Car Market Data Analysis

This repository contains Python code for reading and performing exploratory analysis on the Turkey car market dataset.

## Dataset
The dataset used in this analysis is named `turkey_car_market.csv`. It contains information about various cars in the Turkish market including details like brand, fuel type, transmission, color, body type, condition, displacement (CCM), and horsepower.

## Instructions
1. Ensure you have Python and necessary libraries installed, particularly `pandas` and `numpy`.
2. Download the `turkey_car_market.csv` file and place it in the `dataset` directory within your project folder.
3. Execute the provided Python script to read and analyze the dataset.

## Code Description
- `unprocessed_data = pd.read_csv("../dataset/turkey_car_market.csv")`: Reads the dataset into a pandas DataFrame.
- `unprocessed_data.shape`: Displays the shape of the dataset.
- `unprocessed_data.dtypes`: Displays the data types of columns in the dataset.
- `unprocessed_data.describe()`: Generates descriptive statistics of the dataset.
- `unprocessed_data.drop(["İlan Tarihi","Arac Tip", "Kimden", "Arac Tip Grubu"], axis=1, inplace=True)`: Drops irrelevant columns from the dataset.
- `unprocessed_data.head()`: Displays the first few rows of the dataset.
- Various transformation and cleaning steps are performed on the data including handling missing values, removing non-numeric characters, and converting data types.
- Finally, unique values for specific columns are displayed.

## Analysis
The code provided here constitutes the initial steps of data preprocessing and exploratory analysis. Further analysis and modeling can be performed based on the cleaned dataset obtained from these initial steps.

## Exploratory Data Analysis (EDA)

In this section, exploratory data analysis (EDA) is performed on the processed dataset (`data_processed`). The following steps are carried out:

### Visualizations

- **Countplots**: Count plots are generated for each column in the dataset to visualize the distribution of categorical variables.
- **Distribution Plots**: Distribution plots (displots) are created for numerical variables 'Fiyat' (Price), 'Km', and 'Model Yıl' (Model Year) to understand their distributions.
- **Correlation Analysis**: A heatmap is plotted to visualize the correlation between numerical variables ('Fiyat', 'Km', 'Model Yıl').

### Outlier Removal

- **Outliers Handling**: Outliers are identified and removed from the dataset to ensure the robustness of the analysis. 

### Encoding Categorical Variables

- **Label Encoding**: Categorical variables such as 'Marka' (Brand), 'Yakıt Turu' (Fuel Type), 'Vites' (Transmission), 'Renk' (Color), 'Kasa Tipi' (Body Type), and 'Durum' (Condition) are encoded using label encoding technique.

### Data Processing

- **Data Preparation**: The processed data is reset and unnecessary index column is dropped from the dataset.

### Correlation Analysis

- **Correlation Heatmap**: A correlation heatmap is generated to visualize the correlation between all variables in the processed dataset.

## Usage

To replicate the exploratory data analysis (EDA), follow these steps:

1. Ensure you have Python installed on your system.
2. Install the required libraries such as `seaborn` and `matplotlib`.
3. Execute the provided Python script to perform EDA on the processed dataset.



## Model Building and Evaluation

In this section, various machine learning models are trained and evaluated using the processed dataset (`data_processed`). The following steps are carried out:

### Data Splitting

- The dataset is split into training and testing sets using train-test split method.

### Model Evaluation Function

- An evaluation function `evaluate_model()` is defined to compute the Root Mean Squared Error (RMSE), R-squared (R²), and Mean Squared Error (MSE) for each model.

### Models Trained and Evaluated

- **Linear Regression**:
  - Trained a linear regression model (`lr`) and evaluated its performance.
  
- **Random Forest Regressor**:
  - Trained a random forest regressor (`rf`) with hyperparameters: max_depth=50, n_estimators=200, and evaluated its performance.
  
- **K-Nearest Neighbors Regressor**:
  - Trained a K-nearest neighbors regressor (`knn_reg`) with 3 neighbors and evaluated its performance.
  
- **Support Vector Regressor (SVR)**:
  - Trained a support vector regressor (`svr`) with polynomial kernel and evaluated its performance.

- **Artificial Neural Network (ANN)**:
  - Built and trained a neural network model (`model`) using Keras Sequential API with the following architecture:
    - Input layer with 10 neurons (features)
    - Four hidden layers with 250 neurons each, activated by ReLU
    - Output layer with 1 neuron
  - Compiled the model with mean squared error loss and Adam optimizer.
  - Trained the model for 250 epochs.

### Summary

- Summary of each model's architecture and performance metrics are presented.

## Usage

To replicate the model building and evaluation process, follow these steps:

1. Ensure you have Python installed on your system.
2. Install necessary libraries such as `scikit-learn`, `tensorflow`, and `keras`.
3. Execute the provided Python script to train and evaluate different machine learning models on the processed dataset.



