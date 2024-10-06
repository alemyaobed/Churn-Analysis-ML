# Churn Analysis with Machine Learning

This notebook demonstrates the process of analyzing customer churn data and building predictive models using machine learning techniques. The goal is to understand the data, preprocess it, and apply different classification algorithms to predict customer churn.

## 1. Import Libraries and Load Data

In this section, we import the necessary libraries and load the cleaned dataset.

```python
# Import pandas and load the data is already cleaned
import pandas as pd

# Load the data
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
```

## 2. Initial Data Exploration

We will explore the dataset to understand its structure and contents.

```python
# Display the columns of the data in a vertical list
df.columns.to_frame(index=False)
```

### 2.1 Display First and Last Rows

```python
# Display the first few rows of the dataset
df.head()

# Display the last few rows of the dataset
df.tail()
```

### 2.2 Dataset Shape

```python
# Display the shape of the dataset
df.shape
```

## 3. Checking for Missing Values

### 3.1 Empty Strings in Columns

We check for empty string occurrences in each column.

```python
# Check for empty string occurrence in each column
string_columns = df.select_dtypes(include=['object']).columns
df_copy = df.copy()
empty_strings_per_column = df_copy[string_columns].apply(lambda x: (x.str.strip() == "").sum())
print("Occurrences of empty strings per column:")
print(empty_strings_per_column[empty_strings_per_column > 0])  # Show only columns with empty strings
```

### 3.2 Empty Strings in Rows

We also count empty strings in each row.

```python
# Count occurrences of empty strings in each row
empty_strings_per_row = df_copy[string_columns].apply(lambda x: (x.str.strip() == "").sum(), axis=1)
df_copy['Empty_String_Count'] = empty_strings_per_row
rows_with_empty_strings = df_copy[df_copy['Empty_String_Count'] > 0]
print("Rows with empty strings and their counts:")
print(rows_with_empty_strings[['customerID', 'Empty_String_Count']])
del df_copy  # Remove the copy of the dataframe
```

## 4. Data Type and Initial Checks

```python
# Display the data type of each column
df.dtypes
```

### 4.1 Unique Values and Duplicates

```python
# Display the unique values of the 'Churn' column
df['Churn'].unique()

# Check if customerID is unique
df['customerID'].is_unique

# Check if there are any missing values in the dataset
df.isnull().sum()

# Check if there are any duplicate rows in the dataset
df.duplicated().sum()

# Check the number of unique values in each column
df.nunique()
```

## 5. Encoding Categorical Variables

### 5.1 Binary Columns Encoding

We check and apply label encoding to binary columns.

```python
# Checking if the columns gender, Partner, Dependents, PhoneService, PaperlessBilling, and Churn are binary
check1 = df[['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']].nunique() == 2
check2 = df['gender'].isin(['Male', 'Female']).all() and not df['gender'].isin(['male', 'female']).any()
check3 = df[['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']].isin(['Yes', 'No']).all()

print(check1.all() and check2 and check3.all())
```

Now, we encode these binary columns.

```python
from sklearn.preprocessing import LabelEncoder

# Initialize the label encoder
label_encoder = LabelEncoder()
binary_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']

for col in binary_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Display the first few rows of the updated DataFrame
df.head()
```

### 5.2 One-Hot Encoding for Nominal Variables

We apply one-hot encoding for the nominal variables.

```python
# Using One-Hot Encoding for nominal variables
nominal_columns = ['MultipleLines', 'InternetService', 'OnlineSecurity', 
                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                   'StreamingTV', 'StreamingMovies', 'Contract', 
                   'PaymentMethod']

df = pd.get_dummies(df, columns=nominal_columns, drop_first=True)
df.head()
```

## 6. Data Cleaning

We need to clean the `TotalCharges` column.

```python
# since the TotalCharges has empty strings, we need to drop them and convert the column to float
df = df[df['TotalCharges'].str.strip() != ""]
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
```

## 7. Summary Statistics

```python
# Display the summary statistics of the dataset
df.describe()

# Dropping the customerID column
df.drop(columns=['customerID'], inplace=True)
df.isna().sum()
```

## 8. Data Splitting for Modeling

We will split the data into features and target variable.

```python
from sklearn.model_selection import train_test_split

# Splitting the dataset into features and target variable
X = df.drop('Churn', axis=1)  # Features
y = df['Churn']                # Target variable

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 9. Logistic Regression Model

We will create and evaluate a logistic regression model.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Creating and training the model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Making predictions
y_pred = lr_model.predict(X_test)

print('Evaluation of Logistic Regression Model\n')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Calculating the accuracy of the model
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Accuracy: {accuracy:.2f}%")

# Save the logistic regression model
joblib.dump(lr_model, 'logistic_regression_model.pkl')
```

## 10. Decision Tree Model

Similarly, we will create and evaluate a decision tree model.

```python
from sklearn.tree import DecisionTreeClassifier

# Create a Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Make predictions
y_pred_dt = dt_classifier.predict(X_test)

print('Evaluation of Decision Tree Model\n')
print(confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# Accuracy score
dt_accuracy = accuracy_score(y_test, y_pred_dt) * 100
print(f"Accuracy: {dt_accuracy:.2f}%")

# Save the decision tree model
joblib.dump(dt_classifier, 'decision_tree_model.pkl')
```

## Conclusion

This notebook demonstrates how to preprocess customer churn data, build and evaluate predictive models, and save the trained models for future use. The logistic regression and decision tree models provide insights into customer churn and can help businesses take actionable steps to improve customer retention.
