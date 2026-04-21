# Classificador de Risco — Câncer do Colo do Útero

Trabalho do **Tech Challenge Fase 1** da Pós-Graduação **IA Para Desenvolvedores** — FIAP

---

## Equipe

| Nome | E-mail |
|------|--------|
| Jefferson Antônio Pantoja Silva | jeffkd35@gmail.com |
| Wilson Lima da Silva | wilson.slima@gmail.com |
| Gustavo Lopes da Silva | gustavo_lsilva@hotmail.com |
| Felipe Soeiro Lopes | felipesoeiro.contato@outlook.com.br |
| Vinicius Tavares Sousa da Silva | viniciustavares2014@gmail.com |

---

## Descrição do Projeto

Projeto de aprendizado de máquina para **triagem de pacientes com risco de câncer do colo do útero**.  
O modelo utiliza o dataset *Risk Factors Cervical Cancer*, coletado no Hospital Universitário de Caracas, Venezuela, disponível no UCI Machine Learning Repository.

O objetivo é classificar se uma paciente tem resultado positivo na **Biopsia** (target principal), com base em informações demográficas, comportamentais e histórico médico.

---

## Dataset

| Atributo | Valor |
|----------|-------|
| Nome | Risk Factors for Cervical Cancer |
| Fonte | UCI Machine Learning Repository |
| Link | https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors |
| Arquivo local | `risk_factors_cervical_cancer.csv` |
| Tamanho | 858 pacientes × 36 variáveis |
| Classe positiva (câncer) | 55 pacientes (6%) |
| Classe negativa (saudável) | 803 pacientes (94%) |

> O dataset é altamente **desbalanceado**: apenas 6% dos registros correspondem a casos positivos de biópsia.

### Variáveis originais

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

## Pipeline de Pré-Processamento

O notebook implementa um pipeline sklearn com as seguintes etapas:

1. **Remoção de colunas sem variabilidade** — `STDs:AIDS` e `STDs:cervical condylomatosis` (todos os valores são zero).
2. **Tratamento de valores nulos com indicador de missing** — nulos substituídos pela média + criação de coluna `_preenchido` indicando se a questão foi respondida.
3. **Agrupamento de colunas indicadoras redundantes** — colunas com padrão idêntico de respostas são consolidadas em uma única coluna `_respondido`.
4. **Redução de redundância no tabagismo** — `Smokes (years)` × `Smokes (packs/year)` combinados em `smoke_year_packs`.
5. **Transformação logarítmica de variáveis demográficas** — `Age`, `Number of sexual partners`, `First sexual intercourse`, `Num of pregnancies` (log1p).
6. **Feature engineering de contraceptivos** — criação de `contraceptive_mid_risk` e `contraceptive_exposure`.
7. **Remoção de redundância entre STDs temporais** — `STDs: Time since first diagnosis` removida por ser redundante com `STDs: Time since last diagnosis`.
8. **Criação de atributos estratégicos** — `sexual_risk`, `smoke_age_risk`, `has_std`, `std_age_risk`, `combined_risk`.

---

## Modelo

- **Algoritmo:** Random Forest Classifier (scikit-learn)
- **Configuração:** `class_weight={0: 1, 1: 14}` para compensar o desbalanceamento
- **Divisão:** 80% treino / 20% teste (`stratify=y`, `random_state=42`)

---

## Resultados

```
              precision    recall  f1-score   support

           0       0.94      0.99      0.97       161   (saudável)
           1       0.50      0.09      0.15        11   (câncer)

    accuracy                           0.94       172
   macro avg       0.72      0.54      0.56       172
weighted avg       0.91      0.94      0.91       172
```

| Classe | Precisão | Recall | F1-Score |
|--------|----------|--------|----------|
| Saudável (0) | 94% | 99% | 0.97 |
| Câncer (1) | 50% | 9% | 0.15 |
| **Acurácia geral** | — | — | **94%** |

> **Observação:** O baixo Recall na classe positiva (9%) indica que o modelo ainda perde a maioria dos casos reais de biópsia positiva. Isso é esperado dado o forte desbalanceamento (6% de casos positivos). Estratégias de melhoria incluem SMOTE, ajuste de threshold de decisão e coleta de mais dados rotulados.

---

## Estrutura do Projeto

```
tech-challenge-fase1/
├── risk_factors_cervical_cancer.csv   # Dataset original (UCI)
├── init_cervical_cancer.ipynb         # Notebook principal com análise e modelo
├── relatorio_cancer_cervical.html     # Relatório visual interativo com gráficos
├── relatorio_tecnico_abnt.html        # Relatório técnico completo (formato ABNT, gerar PDF)
└── README.md                          # Este arquivo
```

---

## Instruções de Execução

### Pré-requisitos

- Python 3.8 ou superior
- pip

### 1. Clonar o repositório

```bash
git clone https://github.com/wilsonLima/tech-challenge-fase1
cd tech-challenge-fase1
```

### 2. (Opcional) Criar ambiente virtual

```bash
python -m venv venv

# Linux / Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Instalar as dependências

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 4. Executar o notebook

```bash
jupyter notebook init_cervical_cancer.ipynb
```

Ou, se preferir JupyterLab:

```bash
pip install jupyterlab
jupyter lab init_cervical_cancer.ipynb
```

### 5. Executar todas as células

No Jupyter, clique em **Kernel → Restart & Run All** para executar o pipeline completo do início ao fim.

### 6. Visualizar o relatório HTML

Abra o arquivo `relatorio_cancer_cervical.html` diretamente em qualquer navegador moderno (Chrome, Firefox, Edge).

### 7. Gerar o PDF do relatório técnico (ABNT)

1. Abra `relatorio_tecnico_abnt.html` no Google Chrome ou Edge
2. Pressione `Ctrl + P` (`Cmd + P` no Mac)
3. Selecione **"Salvar como PDF"** como destino
4. Em **Mais configurações**, defina:
   - Tamanho do papel: **A4**
   - Margens: **Nenhuma** (as margens ABNT estão definidas no CSS)
   - Marque **"Gráficos de fundo"** para preservar os fundos coloridos
5. Clique em **Salvar**

---

## Dependências

| Biblioteca | Versão recomendada | Uso |
|------------|-------------------|-----|
| pandas | >= 1.5 | Manipulação e análise de dados |
| numpy | >= 1.23 | Operações numéricas e transformações |
| matplotlib | >= 3.6 | Visualizações e gráficos |
| seaborn | >= 0.12 | Heatmap de correlação e histogramas |
| scikit-learn | >= 1.2 | Pipeline, Random Forest, métricas, split |
| jupyter | >= 1.0 | Ambiente de notebooks interativos |
