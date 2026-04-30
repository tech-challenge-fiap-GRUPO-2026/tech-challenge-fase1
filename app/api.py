#!/usr/bin/env python3

import os

import pickle

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import (
    JSONResponse,
    PlainTextResponse
)
from pydantic import BaseModel
from typing import List
from sklearn.metrics import classification_report

from app.src.constants import (
    MODEL_DIR,
    MODEL_FILE,
    X_TEST_FILE,
    Y_TEST_FILE,
)


app = FastAPI(title="Cervical Cancer Model API")

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

X_TEST_PAH = os.path.join(MODEL_DIR, X_TEST_FILE)

Y_TEST_PAH = os.path.join(MODEL_DIR, Y_TEST_FILE)


model = None
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception:
    print("Ocorreu um erro ao carregar o modelo")
    

X_test = None
try:
    with open(X_TEST_PAH, "rb") as f:
        X_test = pickle.load(f)
except Exception:
    print(
        "Ocorreu um erro ao carregar o dados de teste de X"
    )

y_test = None
try:
    with open(Y_TEST_PAH, "rb") as f:
        y_test = pickle.load(f)
except Exception:
    print(
        "Ocorreu um erro ao carregar o dados de teste de y"
    )


@app.get(
    "/predict",
    summary="Predição do Modelo com Dados de Testes",
    description="Realiza a predição do modelo utilizando os dados utilizados no teste do modelo."
)
async def get_predict_test():
    if model is not None and X_test is not None:
        previsao = model.predict(X_test).tolist()
        return {"prediction": previsao}
    return JSONResponse(
        status_code=500, 
        content={"msg": "Modelo ou dados de Testes de X não podem ser carregados"}
    )


@app.get(
    "/report",
    response_class=PlainTextResponse,
    summary="Relatório de Classificação",
    description="Retorna o relatório de Classificação com base nos Dados de Testes"
)
def report():
    if model is not None and X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        relatorio = classification_report(y_test, y_pred)
        return relatorio
    return JSONResponse(
        status_code=500, 
        content={"msg": "Modelo ou dados de Testes de X e y não podem ser carregados"}
    )
