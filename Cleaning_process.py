# %%
# Import pandas and load the data is already cleaned
import pandas as pd

# Load the data
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')


# %%
# Display the columns of the data in a vertical list
df.columns.to_frame(index=False)


# %%
# Display the first few rows of the dataset
df.head()

# %%
# Display the last few rows of the dataset
df.tail()


# %%
# Display the shape of the dataset
df.shape

# %% [markdown]
# #### Check for empty string occurence in each column

# %%
# Display empty strings in the dataset

# Select only string columns
string_columns = df.select_dtypes(include=['object']).columns

df_copy = df.copy()

# Count occurrences of empty strings in each column
empty_strings_per_column = df_copy[string_columns].apply(lambda x: (x.str.strip() == "").sum())

# Display occurrences of empty strings in each column
print("Occurrences of empty strings per column:")
print(empty_strings_per_column[empty_strings_per_column > 0])  # Show only columns with empty strings


# %% [markdown]
# #### Check for empty string occurence in each row

# %%
# Count occurrences of empty strings in each row
empty_strings_per_row = df_copy[string_columns].apply(lambda x: (x.str.strip() == "").sum(), axis=1)

# Display rows with empty strings and their counts
df_copy['Empty_String_Count'] = empty_strings_per_row  # Add the count as a new column
rows_with_empty_strings = df_copy[df_copy['Empty_String_Count'] > 0]

print("Rows with empty strings and their counts:")
print(rows_with_empty_strings[['customerID', 'Empty_String_Count']])  # Show specific columns and the count

del df_copy  # Remove the copy of the dataframe


# %%
# Display the data type of each column
df.dtypes

# %% [markdown]
# #### Some initial checks

# %%
# Display the unique values of the 'Churn' column
df['Churn'].unique()

# %%
# Check if customerID is unique
df['customerID'].is_unique

# %%
# Check if there are any missing values in the dataset
df.isnull().sum()


# %%
# Check if there are any duplicate rows in the dataset
df.duplicated().sum()

# %%
# Check the number of unique values in each column
df.nunique()

# %% [markdown]
# ####  Checking if the columns  gender, Partner, Dependents, PhoneService, PaperlessBilling, and Churn are binary and applying LabelEncoding to them for optimization.

# %%
# Checking if the columns  gender, Partner, Dependents, PhoneService, PaperlessBilling, and Churn are binary
check1 = df[['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']].nunique() == 2

# Check if the values in Gender are Male or Female strictly
check2 = df['gender'].isin(['Male', 'Female']).all() and not df['gender'].isin(['male', 'female']).any()

# Check if Partner, Dependents, PhoneService, PaperlessBilling, and Churn columns have values Yes or No strictly
check3 = df[['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']].isin(['Yes', 'No']).all()

# if all the checks are passed, print True
print(check1.all() and check2 and check3.all())



# %%
from sklearn.preprocessing import LabelEncoder

# Initialize the label encoder
label_encoder = LabelEncoder()

# Columns with binary values to encode
binary_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']

for col in binary_columns:
    # Encoding details:
    # For binary columns:
    # Yes -> 1
    # No -> 0
    # For gender:
    # Male -> 1
    # Female -> 0
    df[col] = label_encoder.fit_transform(df[col])

# Display the first few rows of the updated DataFrame
df.head()


# %%
# Using One-Hot Encoding for nominal variables
df = pd.get_dummies(df, columns=['MultipleLines', 'InternetService',
                                  'OnlineSecurity', 'OnlineBackup', 
                                  'DeviceProtection', 'TechSupport', 
                                  'StreamingTV', 'StreamingMovies', 
                                  'Contract', 'PaymentMethod'], drop_first=True)

df.head()


# %%
# since the TotalCharges has empty strings, we need to drop them and convert the column to float

# Drop rows with empty strings in the 'TotalCharges' column
df = df[df['TotalCharges'].str.strip() != ""]

# Convert the 'TotalCharges' column to float
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])



# %%
# Display the data type of TotalCharges now
df['TotalCharges'].dtype

# %%
# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df.dtypes


# %%
# Display the summary statistics of the dataset
df.describe()


# %%
# Dropping the customerID column
df.drop(columns=['customerID'], inplace=True)
df.isna().sum()


# %% [markdown]
# ### General splitting for models

# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Splitting the dataset into features and target variable
X = df.drop('Churn', axis=1)  # Features
y = df['Churn']                # Target variable

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %% [markdown]
# #### Logistic Regression Model

# %%
from sklearn.linear_model import LogisticRegression

# Creating and training the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Calculating the accuracy of the model
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Accuracy: {accuracy:.2f}%")


# %% [markdown]
# #### Decision Tree Model

# %%
from sklearn.tree import DecisionTreeClassifier

# Create a Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Make predictions
y_pred_dt = dt_classifier.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# Accuracy score
dt_accuracy = accuracy_score(y_test, y_pred_dt) * 100
print(f"Accuracy: {dt_accuracy:.2f}%")




# %%
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))
plot_tree(dt_classifier, filled=True, feature_names=X.columns)
plt.show()



