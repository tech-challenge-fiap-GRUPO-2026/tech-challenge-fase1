#!/usr/bin/env python3

import os

import pickle

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import List
from sklearn.metrics import classification_report

from app.src.constants import (
    MODEL_DIR,
    MODEL_FILE,
    X_TEST_FILE,
    Y_TEST_FILE
)


app = FastAPI(title="Cervical Cancer Model API")

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

X_TEST_PAH = os.path.join(MODEL_DIR, X_TEST_FILE)

Y_TEST_PAH = os.path.join(MODEL_DIR, Y_TEST_FILE)


model = None
with open(MODEL_PATH, "rb") as f:
    try:
        model = pickle.load(f)
    except Exception:
        print("Ocorreu um erro ao carregar o modelo")
    

X_test = None
with open(X_TEST_PAH, "rb") as f:
    try:
        X_test = pickle.load(f)
    except Exception:
        print(
            "Ocorreu um erro ao carregar o dados de teste de X"
        )

y_test = None
with open(Y_TEST_PAH, "rb") as f:
    try:
        y_test = pickle.load(f)
    except Exception:
        print(
            "Ocorreu um erro ao carregar o dados de teste de y"
        )


class Features(BaseModel):
    features: List[List[float]]


@app.get(
    "/predict",
    summary="Download do modelo treinado"
)
async def get_predict(data: Features):
    if model:
        previsao = model.predict(data.features).tolist()
        return {"prediction": previsao}
    return {"erro": "Modelo não pode ser carregado"}


@app.get(
    "/report",
    response_class=PlainTextResponse
)
def report():
    if model and X_test and y_pred:
        y_pred = model.predict(X_test)
        relatorio = classification_report(y_test, y_pred)
        return relatorio
    return "Modelo ou dados de Testes de X e y não podem ser carregados."
