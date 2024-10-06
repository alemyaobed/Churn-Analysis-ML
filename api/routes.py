from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from .preprocessing import preprocess_input
from .models import predict_churn as make_prediction

router = APIRouter()

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@router.post("/predict")
async def predict_churn(data: CustomerData):
    try:
        # Preprocess input data
        input_data = preprocess_input(data)
        
        # Make prediction
        predictions = make_prediction(input_data)

        # Convert numeric predictions to class labels
        logistic_churn = "Yes" if predictions["Logistic Regression"] == 1 else "No"
        decision_churn = "Yes" if predictions["Decision Tree"] == 1 else "No"

        return {
            "Logistic Regression Churn Prediction": logistic_churn,
            "Decision Tree Churn Prediction": decision_churn
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
