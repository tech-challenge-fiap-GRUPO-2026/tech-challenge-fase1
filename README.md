<div align="center">

# 🎗️ Classificador de Risco — Câncer do Colo do Útero

**Tech Challenge Fase 1 · Pós-Graduação IA Para Desenvolvedores · FIAP**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.99%2B-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E?logo=scikitlearn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

</div>

---

## 👥 Equipe

| Nome | E-mail |
|------|--------|
| Jefferson Antônio Pantoja Silva | jeffkd35@gmail.com |
| Wilson Lima da Silva | wilson.slima@gmail.com |
| Gustavo Lopes da Silva | gustavo_lsilva@hotmail.com |
| Felipe Soeiro Lopes | felipesoeiro.contato@outlook.com.br |
| Vinicius Tavares Sousa da Silva | viniciustavares2014@gmail.com |

---

## 📋 Visão Geral

Projeto de aprendizado de máquina para **triagem de pacientes com risco de câncer do colo do útero**.

O modelo é baseado no dataset *Risk Factors Cervical Cancer*, coletado no Hospital Universitário de Caracas (Venezuela) e disponível no [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors). O objetivo é classificar se uma paciente terá resultado positivo na **Biopsia** (target principal), a partir de informações demográficas, comportamentais e histórico médico.

---

## 📊 Dataset

| Atributo | Valor |
|----------|-------|
| Nome | Risk Factors for Cervical Cancer |
| Fonte | UCI Machine Learning Repository |
| Link | https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors |
| Arquivo local | `data/risk_factors_cervical_cancer.csv` |
| Tamanho | 858 pacientes × 36 variáveis |
| Classe positiva — câncer | 55 pacientes **(6%)** |
| Classe negativa — saudável | 803 pacientes **(94%)** |

> ⚠️ **Dataset altamente desbalanceado:** apenas 6% dos registros correspondem a casos positivos de biópsia.

### 🔬 Variáveis originais

| Variável | Tipo | Descrição |
|----------|:----:|-----------|
| Age | int | Idade da paciente |
| Number of sexual partners | int | Número de parceiros sexuais |
| First sexual intercourse | int | Idade na primeira relação sexual |
| Num of pregnancies | int | Número de gestações |
| Smokes | bool | Fumante |
| Smokes (years) | int | Anos como fumante |
| Smokes (packs/year) | int | Maços por ano |
| Hormonal Contraceptives | bool | Usa anticoncepcional hormonal |
| Hormonal Contraceptives (years) | int | Anos de uso |
| IUD | bool | Usa DIU (dispositivo intrauterino) |
| IUD (years) | int | Anos de uso do DIU |
| STDs | bool | Possui DST |
| STDs (number) | int | Número de DSTs |
| STDs:condylomatosis | bool | Condilomatose |
| STDs:vaginal condylomatosis | bool | Condilomatose vaginal |
| STDs:vulvo-perineal condylomatosis | bool | Condilomatose vulvo-perineal |
| STDs:syphilis | bool | Sífilis |
| STDs:pelvic inflammatory disease | bool | Doença inflamatória pélvica |
| STDs:genital herpes | bool | Herpes genital |
| STDs:molluscum contagiosum | bool | Molusco contagioso |
| STDs:HIV | bool | HIV |
| STDs:Hepatitis B | bool | Hepatite B |
| STDs:HPV | bool | HPV |
| STDs: Number of diagnosis | int | Número de diagnósticos de DST |
| STDs: Time since first diagnosis | int | Tempo desde o primeiro diagnóstico |
| STDs: Time since last diagnosis | int | Tempo desde o último diagnóstico |
| **Hinselmann** | bool | Target *(removido — exame auxiliar)* |
| **Schiller** | bool | Target *(removido — exame auxiliar)* |
| **Citology** | bool | Target *(removido — exame auxiliar)* |
| **Biopsy** | bool | ✅ **Target principal** |
| Dx:Cancer | bool | Diagnóstico prévio *(removido)* |
| Dx:CIN | bool | Diagnóstico prévio *(removido)* |
| Dx:HPV | bool | Diagnóstico prévio *(removido)* |
| Dx | bool | Diagnóstico prévio *(removido)* |

---

## ⚙️ Pipeline de Pré-Processamento

O notebook implementa um pipeline `sklearn` com as seguintes etapas em sequência:

| # | Etapa | Descrição |
|---|-------|-----------|
| 1 | 🗑️ Remoção de colunas sem variabilidade | `STDs:AIDS` e `STDs:cervical condylomatosis` (todos os valores são zero) |
| 2 | 🩹 Tratamento de nulos com indicador de missing | Nulos substituídos pela média + coluna `_preenchido` indicando se a questão foi respondida |
| 3 | 🔗 Agrupamento de indicadoras redundantes | Colunas com padrão idêntico de respostas consolidadas em uma única coluna `_respondido` |
| 4 | 🚬 Redução de redundância no tabagismo | `Smokes (years)` × `Smokes (packs/year)` combinados em `smoke_year_packs` |
| 5 | 📐 Transformação logarítmica demográfica | `Age`, `Number of sexual partners`, `First sexual intercourse`, `Num of pregnancies` (log1p) |
| 6 | 💊 Feature engineering de contraceptivos | Criação de `contraceptive_mid_risk` e `contraceptive_exposure` |
| 7 | ✂️ Remoção de redundância temporal em STDs | `STDs: Time since first diagnosis` removida por ser redundante com `STDs: Time since last diagnosis` |
| 8 | 🧠 Atributos estratégicos | `sexual_risk`, `smoke_age_risk`, `has_std`, `std_age_risk`, `combined_risk` |

---

## 🤖 Modelos Avaliados

Todos os modelos utilizaram divisão **80% treino / 20% teste** com `stratify=y` e `random_state=42` para garantir reprodutibilidade.

---

### Modelo 1 — Random Forest Baseline

Treinado com tratamento mínimo de nulos (apenas imputação pela média), sem o pipeline completo de pré-processamento.

```python
RandomForestClassifier(
    class_weight={0: 1, 1: 10},    # penaliza 10× mais erros na classe positiva (câncer)
    n_estimators=200,
    min_samples_leaf=1,
    random_state=42
)
```

**Busca de hiperparâmetros (GridSearchCV):**

```python
param_grid = {
    'model__n_estimators':     [200, 400],
    'model__max_depth':        [None, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__class_weight':     [{0:1, 1:10}, {0:1, 1:20}]
}
GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
```

---

### Modelo 2 — Random Forest + SMOTE + Threshold ajustado ⭐ Melhor resultado

Pipeline de pré-processamento completo + SMOTE para balancear as classes antes do treino. O limiar de decisão foi ajustado manualmente para maximizar o Recall.

```python
# Oversampling da classe minoritária
SMOTE(random_state=42)

# Classificador
RandomForestClassifier(
    class_weight={0: 1, 1: 10},    # SMOTE + class_weight combinados melhoram o F1
    n_estimators=200,
    min_samples_leaf=1,
    random_state=42
)

# Ajuste do threshold (padrão = 0.50)
threshold = 0.41
y_pred = (y_proba > threshold).astype(int)
```

> 💡 O SMOTE sozinho **piorou** o F1. A combinação SMOTE + `class_weight` foi o que trouxe ganho real.

---

### Modelo 3 — Árvore de Decisão (sem SMOTE)

Pipeline completo de pré-processamento, porém **sem SMOTE** — experimentos mostraram que a Árvore de Decisão funciona melhor somente com o `class_weight` penalizado, sem oversampling sintético.

```python
DecisionTreeClassifier(
    class_weight={0: 1, 1: 10},    # penaliza erros na classe positiva
    max_depth=4,                   # limita profundidade para evitar overfitting
    min_samples_leaf=5,            # folhas com pelo menos 5 amostras
    random_state=42
)
```

> Threshold padrão de 0.50 mantido — ajustes não trouxeram ganho neste modelo.

---

## 📈 Resultados Comparativos

| Modelo | Precisão (câncer) | Recall (câncer) | F1 (câncer) | Acurácia | AUROC |
|--------|:-----------------:|:---------------:|:-----------:|:--------:|:-----:|
| RF Baseline | 50% | 9% | 0.15 | 94% | — |
| **RF + SMOTE + Threshold 0.41** ⭐ | **31%** | **36%** | **0.33** | **91%** | **0.737** |
| Árvore de Decisão (sem SMOTE) | 22% | 36% | 0.28 | 88% | 0.624 |

> 🏥 **Recall** é a métrica mais importante neste contexto clínico — representa a capacidade de identificar corretamente os casos de câncer, minimizando falsos negativos.

**Conclusão:** O melhor resultado foi obtido com **Random Forest + SMOTE + threshold 0.41**, elevando o Recall de 9% para 36%. O pré-processamento melhorou a importância das features, mas o dataset possui baixo poder preditivo intrínseco — é pequeno (858 pacientes) e altamente desbalanceado (6% positivos). O uso de SMOTE, embora eficaz tecnicamente, não é bem-visto pela comunidade médica. A Árvore de Decisão, mesmo com AUROC inferior (0.624), pode ser preferível em contextos que exigem interpretabilidade ou que rejeitem oversampling sintético.

---

## 📁 Estrutura do Projeto

```
tech-challenge-fase1/
│
├── data/
│   └── risk_factors_cervical_cancer.csv    # Dataset original (UCI)
│
├── notebooks/
│   └── init_cervical_cancer.ipynb          # Notebook principal com análise e modelos
│
├── docs/
│   ├── relatorio_tecnico_abnt.pdf          # Relatório técnico completo (PDF — ABNT)
│   └── relatorio_tecnico_abnt.docx         # Relatório técnico completo (Word)
│
├── app/
│   ├── __init__.py
│   ├── api.py                              # Ponto de entrada da API (FastAPI)
│   ├── training.py                         # Ponto de entrada do treinamento
│   └── src/
│       ├── __init__.py
│       ├── constants.py                    # Constantes compartilhadas
│       ├── data.py                         # Carregamento e tratamento do dataset
│       ├── pipeline_functions.py           # Funções de pré-processamento
│       ├── streamlit_helper.py             # Componente auxiliar legado (não usado pela API)
│       └── train_test.py                   # Treinamento e avaliação dos modelos
│
├── model/                                  # Gerado automaticamente pelo run_training.sh
│   ├── model.pkl                           # Modelo treinado serializado
│   ├── X_test.pkl                          # Dados de teste (features)
│   └── y_test.pkl                          # Dados de teste (labels)
│
├── Dockerfile                              # Imagem Docker da aplicação
├── docker-compose.yml                      # Orquestração dos containers
├── requirements.txt                        # Dependências Python
├── run_api.sh                              # Script de execução da API
├── run_training.sh                         # Script de treinamento e deploy do modelo
├── relatorio_cancer_cervical.html          # Relatório visual interativo com gráficos
└── README.md                               # Este arquivo
```

---

## 🚀 Instruções de Execução

### ✅ Pré-requisitos

- Python 3.8 ou superior
- pip
- Docker *(opcional, para execução em container)*

---

### 1. 📥 Clonar o repositório

```bash
git clone https://github.com/wilsonLima/tech-challenge-fase1
cd tech-challenge-fase1
```

### 2. 🐍 Criar ambiente virtual *(opcional)*

```bash
python -m venv venv

# Linux / Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. 📦 Instalar as dependências

```bash
pip install -r requirements.txt
```

### 4. ▶️ Executar o notebook

```bash
jupyter notebook notebooks/init_cervical_cancer.ipynb
```

Ou, se preferir JupyterLab:

```bash
jupyter lab notebooks/init_cervical_cancer.ipynb
```

No Jupyter, clique em **Kernel → Restart & Run All** para executar o pipeline completo do início ao fim.

---

### 5. 🖥️ Executar a API com Modelo Treinado

**Via script (Linux):**

```bash
# Treinar o modelo e fazer deploy no diretório model/
./run_training.sh

# Subir a API com o modelo
./run_api.sh
```

**Via Docker *(recomendado)*:**

```bash
# Gerar a imagem com o modelo treinado
docker compose build --no-cache

# Subir a API
docker compose up -d

# Parar a API
docker compose down
```

Acesse a documentação interativa em: **`http://localhost:8000/docs`**

---

### 6. 🌐 Visualizar o relatório HTML

Abra o arquivo `relatorio_cancer_cervical.html` diretamente em qualquer navegador moderno (Chrome, Firefox, Edge).

### 7. 📄 Relatório técnico (ABNT)

O relatório técnico está disponível em dois formatos:

- `docs/relatorio_tecnico_abnt.pdf` — abra diretamente em qualquer leitor de PDF.
- `docs/relatorio_tecnico_abnt.docx` — abra no Microsoft Word ou Google Docs para edição.

---

## 🌐 Documentação da API

A API é disponibilizada via **FastAPI** com documentação interativa automática gerada pelo Swagger.

| Interface | URL |
|-----------|-----|
| Swagger UI | `http://localhost:8000/docs` |
| ReDoc | `http://localhost:8000/redoc` |

**Base URL:** `http://localhost:8000`

---

### Endpoints

| Método | Rota | Resposta | Descrição |
|:------:|------|----------|-----------|
| `GET` | `/predict` | `application/json` | Executa predição com os dados de teste |
| `GET` | `/report` | `text/plain` | Retorna o relatório de classificação |

---

#### `GET /predict`

Realiza a predição utilizando os dados de teste (`X_test.pkl`) serializados em `model/` durante o treinamento.

**200 OK**

```json
{
  "prediction": [0, 0, 1, 0, 0, 1]
}
```

> `0` = Biópsia negativa (saudável) · `1` = Biópsia positiva (câncer)

**500 Internal Server Error**

```json
{
  "msg": "Modelo ou dados de Testes de X não podem ser carregados"
}
```

---

#### `GET /report`

Retorna o relatório de classificação completo gerado pelo `scikit-learn` com base nos dados de teste (`X_test.pkl` e `y_test.pkl`).

**200 OK** — `text/plain`

```
              precision    recall  f1-score   support

           0       0.96      0.94      0.95       161
           1       0.31      0.36      0.33        11

    accuracy                           0.91       172
   macro avg       0.63      0.65      0.64       172
weighted avg       0.92      0.91      0.91       172
```

**500 Internal Server Error**

```json
{
  "msg": "Modelo ou dados de Testes de X e y não podem ser carregados"
}
```

---

## 🧩 Dependências

| Biblioteca | Versão | Uso |
|------------|:------:|-----|
| scipy | >= 1.10 | Computação científica |
| pandas | >= 1.5 | Manipulação e análise de dados |
| numpy | >= 1.23 | Operações numéricas e transformações |
| matplotlib | >= 3.6 | Visualizações e gráficos |
| seaborn | >= 0.12 | Heatmap de correlação e histogramas |
| scikit-learn | >= 1.2 | Pipeline, Random Forest, Decision Tree, métricas, GridSearchCV |
| imbalanced-learn | >= 0.10 | SMOTE — oversampling da classe minoritária |
| notebook | >= 1.0 | Jupyter clássico |
| jupyterlab | >= 3.6.8 | Jupyter moderno |
| fastapi | >= 0.99.1 | Framework da API REST |
| uvicorn | >= 0.30.6 | Servidor ASGI para a API |

---

## 🎬 Vídeo Demonstração

[![Assistir no YouTube](https://img.shields.io/badge/YouTube-Assistir%20Demo-FF0000?logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=Pa4ZjERYIrQ)

```
https://www.youtube.com/watch?v=Pa4ZjERYIrQ
```
