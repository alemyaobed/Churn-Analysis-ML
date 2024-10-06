# Customer Churn Prediction

## Overview

This project implements a machine learning pipeline to predict customer churn for a telecommunications company. The dataset is processed, cleaned, and transformed to be suitable for model training. Two models, Logistic Regression and Decision Tree, are built and evaluated for their performance in predicting churn.

## Table of Contents

- [Installation and Usage](#installation-and-usage)
- [Data Description](#data-description)
- [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [API Documentation](#api-documentation)
- [License](#license)

## Installation and Usage

The model and api are available in the repo along with the data set, so you could start the whole training process on your own or use the models or test using the api: by looking at samples of the data set.

1. Clone the repository:

   ```bash
   git clone https://github.com/alemyaobed/Churn-Analysis-ML.git
   cd Churn-Analysis-ML
   ```

2. Install the needed packages and libraries

    ```bash
    # Create a virtual environment
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`

    # Install required packages
    pip install requirements.txt
    ```

3. **Running the API**:

   To run the API, execute the following command:

   ```bash
   uvicorn main:app --reload
   ```

   This will start the API server, and you can access it at `http://localhost:8000` (or the specified port).

4. **Testing using the API**:

   To test the API, refer to the [API Documentation](#api-documentation) section for details on how to make requests.

## Data Description

Wanting to do your own? Check [the_whole_process.md](the_whole_process.md).
The dataset consists of various features related to customer information, such as demographic details, account information, and services used, with the target variable being whether the customer has churned.

## Data Cleaning and Preprocessing

The following steps were taken to clean and preprocess the data:

1. **Load the data**:

    Ensure you have the dataset `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the same directory as your script.

    ```python
    import pandas as pd
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    ```

2. **Check for empty strings**:

   ```python
   empty_strings_per_column = df_copy[string_columns].apply(lambda x: (x.str.strip() == "").sum())
   ```

3. **Label Encoding for binary variables**:

   ```python
   from sklearn.preprocessing import LabelEncoder
   label_encoder = LabelEncoder()
   binary_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
   ```

4. **One-Hot Encoding for nominal variables**:

   ```python
   df = pd.get_dummies(df, columns=nominal_columns, drop_first=True)
   ```

5. **Convert `TotalCharges` to float**:

   ```python
   df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
   ```

## Model Training

Two models were built using the processed data:

### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
```

### Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
```

## Evaluation

Both models were evaluated using metrics like accuracy, confusion matrix, and classification report.

### Example Evaluation Output for Logistic Regression

```plaintext
Evaluation of Logistic Regression Model

[[917 116]
 [186 188]]
              precision    recall  f1-score   support

           0       0.83      0.89      0.86      1033
           1       0.62      0.50      0.55       374

    accuracy                           0.79      1407
   macro avg       0.72      0.70      0.71      1407
weighted avg       0.77      0.79      0.78      1407

Accuracy: 78.54%
```

### Example Evaluation Output for Decision Tree

```plaintext
Evaluation of Decision Tree Model

[[830 203]
 [185 189]]
              precision    recall  f1-score   support

           0       0.82      0.80      0.81      1033
           1       0.48      0.51      0.49       374

    accuracy                           0.72      1407
   macro avg       0.65      0.65      0.65      1407
weighted avg       0.73      0.72      0.73      1407

Accuracy: 72.42%
```

## API Documentation

The API allows users to interact with the customer churn prediction model. Users can submit customer data and receive a prediction on whether the customer is likely to churn.

### Base URL

```curl
http://localhost:8000/
```

### Endpoints

#### 1. Predict Churn

- **URL**: `/predict`
- **Method**: `POST`
- **Description**: Predicts whether a customer will churn based on the provided customer data.

##### Request

- **Headers**:
  - `Content-Type`: application/json
- **Body**:

```json
{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "One year",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Credit card (automatic)",
    "MonthlyCharges": 60.2,
    "TotalCharges": 720.0
}
```

##### Response

- **Success Response**:
  - **Code**: `200 OK`
  - **Content**:

```json
{
    "Logistic Regression Churn Prediction with 78.54% accuracy": "No",
    "Decision Tree Churn Prediction with 72.42% accuracy": "No"
}
```

- **Error Response**:
  - **Code**: `400 Bad Request`
  - **Content**:

```json
{
    "detail": "Invalid input data."
}
```

### Example cURL Request

```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "One year",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Credit card (automatic)",
    "MonthlyCharges": 60.2,
    "TotalCharges": 720.0
}'
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
