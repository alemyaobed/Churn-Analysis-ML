import joblib

# Load the saved models
logistic_model = joblib.load('logistic_regression_model.pkl')
decision_tree_model = joblib.load('decision_tree_model.pkl')

def predict_churn(input_data):
    """
    Predict churn using logistic and decision tree models.
    
    Args:
        input_data (array-like): Preprocessed input data for prediction.
        
    Returns:
        dict: A dictionary containing predictions from both models.
    """
    logistic_prediction = logistic_model.predict(input_data)
    decision_tree_prediction = decision_tree_model.predict(input_data)

    return {
        "Logistic Regression": logistic_prediction[0],
        "Decision Tree": decision_tree_prediction[0]
    }
