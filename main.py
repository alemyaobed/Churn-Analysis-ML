# main.py

from fastapi import FastAPI
from api.routes import router

app = FastAPI()

# Include the API router
app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Churn Prediction API!"}
