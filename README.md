# Predição de Risco — Câncer do Colo do Útero

## Descrição

Projeto de aprendizado de máquina para triagem de pacientes com risco de câncer do colo do útero.
O modelo é treinado sobre o dataset *Risk Factors Cervical Cancer*, coletado no Hospital Universitário
de Caracas, Venezuela, disponível no UCI Machine Learning Repository.

## Dataset

- **Nome:** Risk Factors for Cervical Cancer
- **Fonte:** UCI Machine Learning Repository
- **Link de download:** https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors
- **Arquivo local:** `risk_factors_cervical_cancer.csv`
- **Tamanho:** 858 pacientes × 36 variáveis

## Estrutura do Projeto

```
dataset/
├── risk_factors_cervical_cancer.csv   # Dataset original
├── init_cervical_cancer.ipynb         # Notebook principal com todo o código
├── relatorio_cancer_cervical.html     # Relatório visual interativo (PT-BR)
├── relatorio_tecnico_pdf.html         # Relatório técnico para impressão em PDF
└── README.md                          # Este arquivo
```

## Instruções de Execução

### Pré-requisitos

- Python 3.8 ou superior
- pip

### Instalação das dependências

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Executar o notebook

```bash
jupyter notebook init_cervical_cancer.ipynb
```

Ou, se preferir JupyterLab:

```bash
pip install jupyterlab
jupyter lab init_cervical_cancer.ipynb
```

### Visualizar o relatório HTML

Abra o arquivo `relatorio_cancer_cervical.html` em qualquer navegador moderno (Chrome, Firefox, Edge).

### Gerar o PDF do relatório técnico

1. Abra `relatorio_tecnico_pdf.html` no Google Chrome ou Edge
2. Pressione `Ctrl + P` (ou `Cmd + P` no Mac)
3. Selecione **"Salvar como PDF"** como destino
4. Clique em **Salvar**

## Dependências Utilizadas

| Biblioteca    | Versão recomendada | Uso                                     |
|---------------|--------------------|-----------------------------------------|
| pandas        | >= 1.5             | Manipulação e análise de dados          |
| numpy         | >= 1.23            | Operações numéricas                     |
| matplotlib    | >= 3.6             | Visualizações                           |
| seaborn       | >= 0.12            | Heatmap de correlação                   |
| scikit-learn  | >= 1.2             | Random Forest, métricas, train/test split |
| jupyter       | >= 1.0             | Ambiente de notebooks                   |

## Resumo dos Resultados

| Modelo           | Acurácia | Precisão (Cl. +) | Recall (Cl. +) | F1 (Cl. +) |
|------------------|----------|------------------|----------------|------------|
| RF Baseline      | 94%      | 50%              | 9%             | 0,15       |
| Transfer Learning| 94%      | 50%              | 9%             | 0,15       |

> ⚠️ O baixo Recall na classe positiva indica que o modelo ainda perde a maioria dos casos
> reais de biópsia positiva. Estratégias de melhoria incluem SMOTE, ajuste de threshold e
> mais dados rotulados.
