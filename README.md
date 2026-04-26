# 🎗️ Classificador de Risco — Câncer do Colo do Útero

Trabalho do **Tech Challenge Fase 1** da Pós-Graduação **IA Para Desenvolvedores** — FIAP

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

## 📋 Descrição do Projeto

Projeto de aprendizado de máquina para **triagem de pacientes com risco de câncer do colo do útero**.  
O modelo utiliza o dataset *Risk Factors Cervical Cancer*, coletado no Hospital Universitário de Caracas, Venezuela, disponível no UCI Machine Learning Repository.

O objetivo é classificar se uma paciente tem resultado positivo na **Biopsia** (target principal), com base em informações demográficas, comportamentais e histórico médico.

---

## 📊 Dataset

| Atributo | Valor |
|----------|-------|
| Nome | Risk Factors for Cervical Cancer |
| Fonte | UCI Machine Learning Repository |
| Link | https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors |
| Arquivo local | `data/risk_factors_cervical_cancer.csv` |
| Tamanho | 858 pacientes × 36 variáveis |
| Classe positiva (câncer) | 55 pacientes (6%) |
| Classe negativa (saudável) | 803 pacientes (94%) |

> ⚠️ O dataset é altamente **desbalanceado**: apenas 6% dos registros correspondem a casos positivos de biópsia.

### 🔬 Variáveis originais

| Variável | Tipo | Descrição |
|----------|------|-----------|
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
| **Hinselmann** | bool | Target (removido — exame auxiliar) |
| **Schiller** | bool | Target (removido — exame auxiliar) |
| **Citology** | bool | Target (removido — exame auxiliar) |
| **Biopsy** | bool | **Target principal** |
| Dx:Cancer | bool | Diagnóstico prévio (removido) |
| Dx:CIN | bool | Diagnóstico prévio (removido) |
| Dx:HPV | bool | Diagnóstico prévio (removido) |
| Dx | bool | Diagnóstico prévio (removido) |

---

## ⚙️ Pipeline de Pré-Processamento

O notebook implementa um pipeline sklearn com as seguintes etapas:

1. 🗑️ **Remoção de colunas sem variabilidade** — `STDs:AIDS` e `STDs:cervical condylomatosis` (todos os valores são zero).
2. 🩹 **Tratamento de valores nulos com indicador de missing** — nulos substituídos pela média + criação de coluna `_preenchido` indicando se a questão foi respondida.
3. 🔗 **Agrupamento de colunas indicadoras redundantes** — colunas com padrão idêntico de respostas são consolidadas em uma única coluna `_respondido`.
4. 🚬 **Redução de redundância no tabagismo** — `Smokes (years)` × `Smokes (packs/year)` combinados em `smoke_year_packs`.
5. 📐 **Transformação logarítmica de variáveis demográficas** — `Age`, `Number of sexual partners`, `First sexual intercourse`, `Num of pregnancies` (log1p).
6. 💊 **Feature engineering de contraceptivos** — criação de `contraceptive_mid_risk` e `contraceptive_exposure`.
7. ✂️ **Remoção de redundância entre STDs temporais** — `STDs: Time since first diagnosis` removida por ser redundante com `STDs: Time since last diagnosis`.
8. 🧠 **Criação de atributos estratégicos** — `sexual_risk`, `smoke_age_risk`, `has_std`, `std_age_risk`, `combined_risk`.

---

## 🤖 Modelos Avaliados

Foram testados três cenários progressivos de modelagem:

### 1. Random Forest — Baseline (sem oversampling)
- **Algoritmo:** RandomForestClassifier · `class_weight={0: 1, 1: 14}`
- **Divisão:** 80% treino / 20% teste · `stratify=y` · `random_state=42`

### 2. Random Forest — Com SMOTE + Ajuste de Threshold ⭐ melhor resultado
- **Pipeline:** mesmo pré-processamento + SMOTE (oversampling da classe minoritária)
- **Threshold ajustado:** 0.41 (em vez do padrão 0.50)

### 3. Árvore de Decisão — Com SMOTE
- **Algoritmo:** DecisionTreeClassifier · `class_weight={0: 1, 1: 14}`
- **GridSearchCV** para otimização de hiperparâmetros

---

## 📈 Resultados Comparativos

| Modelo | Precisão (câncer) | Recall (câncer) | F1 (câncer) | Acurácia | AUROC |
|--------|:-----------------:|:---------------:|:-----------:|:--------:|:-----:|
| RF Baseline | 50% | 9% | 0.15 | 94% | — |
| **RF + SMOTE + Threshold 0.41** | **31%** | **36%** | **0.33** | **91%** | **0.737** |
| Árvore de Decisão + SMOTE | 22% | 36% | 0.28 | 88% | 0.624 |

> 💡 **Conclusão:** O melhor resultado foi obtido com **Random Forest + SMOTE + threshold ajustado (0.41)**, que elevou o Recall de 9% para 36%. O dataset possui baixo poder preditivo intrínseco — o pré-processamento melhorou a importância das features, mas não a capacidade discriminatória do modelo. O uso de SMOTE, embora eficaz tecnicamente, não é bem-visto pela comunidade médica. A Árvore de Decisão apresentou AUROC inferior (0.62), confirmando o Random Forest como algoritmo mais adequado para este problema.

---

## 📁 Estrutura do Projeto

```
tech-challenge-fase1/
├── data/
│   └── risk_factors_cervical_cancer.csv   # Dataset original (UCI)
├── notebooks/
│   └── init_cervical_cancer.ipynb         # Notebook principal com análise e modelos
├── reports/
│   ├── relatorio_cancer_cervical.html     # Relatório visual interativo com gráficos
│   ├── relatorio_tecnico_abnt.pdf         # Relatório técnico completo (formato ABNT — PDF)
│   └── relatorio_tecnico_abnt.docx        # Relatório técnico completo (formato Word)
├── app/
│   ├── __init__.py
│   ├── main.py                            # Ponto de entrada da aplicação Streamlit
│   └── src/
│       ├── data.py                        # Carregamento e tratamento do dataset
│       ├── pipeline_functions.py          # Funções de pré-processamento do pipeline
│       ├── streamlit_helper.py            # Componentes de UI (Streamlit)
│       └── train_test.py                  # Treinamento e avaliação dos modelos
├── Dockerfile                             # Imagem Docker da aplicação
├── docker-compose.yml                     # Orquestração dos containers
├── requirements.txt                       # Dependências Python
├── run_app.sh                             # Script de execução (Linux/Mac)
├── run_app.cmd                            # Script de execução (Windows)
└── README.md                              # Este arquivo
```

---

## 🚀 Instruções de Execução

### ✅ Pré-requisitos

- Python 3.8 ou superior
- pip

### 1. 📥 Clonar o repositório

```bash
git clone https://github.com/wilsonLima/tech-challenge-fase1
cd tech-challenge-fase1
```

### 2. 🐍 (Opcional) Criar ambiente virtual

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
pip install jupyterlab
jupyter lab notebooks/init_cervical_cancer.ipynb
```

### 5. ⚡ Executar todas as células

No Jupyter, clique em **Kernel → Restart & Run All** para executar o pipeline completo do início ao fim.

### 6. 🖥️ Executar a aplicação Streamlit

**Via script (recomendado):**

```bash
# Linux / Mac
bash run_app.sh

# Windows
run_app.cmd
```

**Via Docker:**

```bash
docker-compose up --build
```

Acesse em: `http://localhost:8501`

### 7. 🌐 Visualizar o relatório HTML

Abra o arquivo `reports/relatorio_cancer_cervical.html` diretamente em qualquer navegador moderno (Chrome, Firefox, Edge).

### 8. 📄 Relatório técnico (ABNT)

O relatório técnico já está disponível no repositório em dois formatos:

- `reports/relatorio_tecnico_abnt.pdf` — abra diretamente em qualquer leitor de PDF.
- `reports/relatorio_tecnico_abnt.docx` — abra no Microsoft Word ou Google Docs para edição.

---

## 🧩 Dependências

| Biblioteca | Versão recomendada | Uso |
|------------|-------------------|-----|
| pandas | >= 1.5 | Manipulação e análise de dados |
| numpy | >= 1.23 | Operações numéricas e transformações |
| matplotlib | >= 3.6 | Visualizações e gráficos |
| seaborn | >= 0.12 | Heatmap de correlação e histogramas |
| scikit-learn | >= 1.2 | Pipeline, Random Forest, Decision Tree, métricas, GridSearchCV |
| imbalanced-learn | >= 0.10 | SMOTE — oversampling da classe minoritária |
| streamlit | >= 1.20 | Interface web interativa da aplicação |
| jupyter | >= 1.0 | Ambiente de notebooks interativos |
