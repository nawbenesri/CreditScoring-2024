from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import general

app = FastAPI()

class CreateDfRequest(BaseModel):
    path: str

class FiltrerColumnsRequest(BaseModel):
    df: List[Dict[str, Any]]
    columns: List[str]

class CleanDataRequest(BaseModel):
    df: List[Dict[str, Any]]

class LogisticRegressionRequest(BaseModel):
    X_test: List[List[float]]
    y_train: List[int]
    X_train: List[List[float]]
    y_test: List[int]
    outcome_path: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI application"}

@app.post("/create_df/")
def create_df(request: CreateDfRequest):
    try:
        df = general.create_df(request.path)
        return {"result": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/filtrer_columns/")
def filtrer_columns(request: FiltrerColumnsRequest):
    try:
        df = pd.DataFrame(request.df)
        general.filtrer_columns(df, request.columns)
        return {"result": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/clean_data/")
def clean_data(request: CleanDataRequest):
    try:
        df = pd.DataFrame(request.df)
        general.clean_data(df)
        return {"result": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/logistic_regression/")
def logistic_regression(request: LogisticRegressionRequest):
    try:
        X_test = np.array(request.X_test)
        y_train = np.array(request.y_train)
        X_train = np.array(request.X_train)
        y_test = np.array(request.y_test)
        result = general.logistic_regression(X_test, y_train, X_train, y_test, request.outcome_path)
        return {"result": result.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
