import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_input(data):
    # Create a DataFrame from input data
    df = pd.DataFrame([data.dict()])
    
    # Drop customerID column if present
    df = df.drop(columns=['customerID'], errors='ignore')

    # Apply label encoding for binary columns
    binary_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']

    # Initialize the label encoder
    label_encoder = LabelEncoder()

    for col in binary_columns:
        # Encoding Yes -> 1, No -> 0 for binary columns
        df[col] = label_encoder.fit_transform(df[col])

    # Manually set the values for nominal columns based on the specific values in data
    
    # MultipleLines
    df['MultipleLines_No phone service'] = 1 if df['MultipleLines'][0] == 'No phone service' else 0
    df['MultipleLines_Yes'] = 1 if df['MultipleLines'][0] == 'Yes' else 0

    # InternetService
    df['InternetService_Fiber optic'] = 1 if df['InternetService'][0] == 'Fiber optic' else 0
    df['InternetService_No'] = 1 if df['InternetService'][0] == 'No' else 0

    # OnlineSecurity
    df['OnlineSecurity_No internet service'] = 1 if df['OnlineSecurity'][0] == 'No internet service' else 0
    df['OnlineSecurity_Yes'] = 1 if df['OnlineSecurity'][0] == 'Yes' else 0

    # OnlineBackup
    df['OnlineBackup_No internet service'] = 1 if df['OnlineBackup'][0] == 'No internet service' else 0
    df['OnlineBackup_Yes'] = 1 if df['OnlineBackup'][0] == 'Yes' else 0

    # DeviceProtection
    df['DeviceProtection_No internet service'] = 1 if df['DeviceProtection'][0] == 'No internet service' else 0
    df['DeviceProtection_Yes'] = 1 if df['DeviceProtection'][0] == 'Yes' else 0

    # TechSupport
    df['TechSupport_No internet service'] = 1 if df['TechSupport'][0] == 'No internet service' else 0
    df['TechSupport_Yes'] = 1 if df['TechSupport'][0] == 'Yes' else 0

    # StreamingTV
    df['StreamingTV_No internet service'] = 1 if df['StreamingTV'][0] == 'No internet service' else 0
    df['StreamingTV_Yes'] = 1 if df['StreamingTV'][0] == 'Yes' else 0

    # StreamingMovies
    df['StreamingMovies_No internet service'] = 1 if df['StreamingMovies'][0] == 'No internet service' else 0
    df['StreamingMovies_Yes'] = 1 if df['StreamingMovies'][0] == 'Yes' else 0

    # Contract
    df['Contract_One year'] = 1 if df['Contract'][0] == 'One year' else 0
    df['Contract_Two year'] = 1 if df['Contract'][0] == 'Two year' else 0

    # PaymentMethod
    df['PaymentMethod_Credit card (automatic)'] = 1 if df['PaymentMethod'][0] == 'Credit card (automatic)' else 0
    df['PaymentMethod_Electronic check'] = 1 if df['PaymentMethod'][0] == 'Electronic check' else 0
    df['PaymentMethod_Mailed check'] = 1 if df['PaymentMethod'][0] == 'Mailed check' else 0

    # Drop original nominal columns after encoding them
    df = df.drop(columns=['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                          'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                          'Contract', 'PaymentMethod'], errors='ignore')

    return df
