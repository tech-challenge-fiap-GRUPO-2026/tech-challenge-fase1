
import pandas as pd
import numpy as np

def remover_colunas_sem_variacao(df):
    df = df.copy()
    cols_remover = ['STDs:AIDS', 'STDs:cervical condylomatosis']
    df.drop(columns=cols_remover, inplace=True)
    return df


def tratar_missing_com_indicador(dataset):
    dt = dataset.copy()
    colunas = dt.columns.tolist()
    for col in colunas:
        if dt[col].isnull().any():
            # Criar coluna indicador (1 = respondeu, 0 = não respondeu)
            nome_nova_coluna = f"{col}_preenchido"
            dt[nome_nova_coluna] = dt[col].notnull().astype(int)
            # Calcular média
            media = dt[col].mean()
            # Preencher nulos
            dt[col] = dt[col].fillna(media)
    return dt


def agrupar_colunas_preenchido(df, sufixo='_preenchido', novo_sufixo='_respondido'):
    df = df.copy()
    # 1. Selecionar colunas com sufixo
    colunas = [col for col in df.columns if sufixo in col]
    # 2. Agrupar colunas idênticas
    grupos = {}
    for col in colunas:
        valores = tuple(df[col].values)

        if valores in grupos:
            grupos[valores].append(col)
        else:
            grupos[valores] = [col]

    novas_colunas = []

    # 3. Criar novas colunas e remover redundantes
    for grupo in grupos.values():

        if len(grupo) > 1:
            nome_base = grupo[0].replace(sufixo, '')
            nova_coluna = f"{nome_base}{novo_sufixo}"

            df[nova_coluna] = df[grupo[0]]
            df.drop(columns=grupo, inplace=True)

            novas_colunas.append(nova_coluna)

        else:
            col = grupo[0]
            novo_nome = col.replace(sufixo, novo_sufixo)

            df.rename(columns={col: novo_nome}, inplace=True)
            novas_colunas.append(novo_nome)

    return df


def reduzir_redundancia_smokes(df):
    df = df.copy()
    df.loc[df['Smokes'] == 0, ['Smokes (years)', 'Smokes (packs/year)']] = 0
    # combinar
    df['smoke_year_packs'] = df['Smokes (years)'] * df['Smokes (packs/year)']
    # remover redundantes
    df.drop(columns=['Smokes (years)', 'Smokes (packs/year)'], inplace=True)
    return df


def transformar_demograficos_comportamentais(df):
    df = df.copy()
    cols = [
        'Age',
        'Number of sexual partners',
        'First sexual intercourse',
        'Num of pregnancies',
    ]
    # Transformação log (só onde faz sentido)
    for col in cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    return df


def discretizar_demograficos_comportamentais(df):
    df = df.copy()
    config = {
        'Age': [0, 20, 30, 40, 100],
        'Number of sexual partners': [0, 2, 5, 10, 50],
        'First sexual intercourse': [0, 14, 17, 20, 40],
        'Num of pregnancies': [-1, 0, 2, 5, 20],
    }
    for col, bins in config.items():
        df[col] = pd.cut(df[col], bins=bins, labels=False, include_lowest=True)
    return df


def transformar_contraceptivos(df):
    df = df.copy()
    df.loc[df['Hormonal Contraceptives'] == 0, 'Hormonal Contraceptives (years)'] = 0
    col_bin = 'Hormonal Contraceptives'
    col_years = 'Hormonal Contraceptives (years)'
    if {col_bin, col_years}.issubset(df.columns):
        # Flag de risco intermediário
        df['contraceptive_mid_risk'] = (
            (df[col_bin] == 1) &
            (df[col_years].between(3, 15))
        ).astype(int)
        # Interação (exposição)
        df['contraceptive_exposure'] = df[col_bin] * df[col_years]
    return df


def remover_coluna_std_tempo(df):
    df = df.copy()
    df.drop(columns=['STDs: Time since first diagnosis'], inplace=True)
    return df


def remover_coluna_std_tempo(df):
    df = df.copy()
    df.drop(columns=['STDs: Time since first diagnosis'], inplace=True)
    return df


def criar_atributos_estrategicos(df):
    df = df.copy()
    # RISCO SEXUAL (comportamento + idade)
    if {'Number of sexual partners', 'Age'}.issubset(df.columns):
        df['sexual_risk'] = df['Number of sexual partners'] / (df['Age'] + 1)
    # TABAGISMO
    if 'smoke_year_packs' in df.columns:

        if 'Age' in df.columns:
            df['smoke_age_risk'] = df['smoke_year_packs'] / (df['Age'] + 1)
    # STDs (raras → simplificar)
    if 'STDs (number)' in df.columns:
        df['has_std'] = (df['STDs (number)'] > 0).astype(int)

        if 'Age' in df.columns:
            df['std_age_risk'] = df['STDs (number)'] / (df['Age'] + 1)

    # RISCO COMBINADO
    if {
        'Number of sexual partners',
        'Hormonal Contraceptives',
        'STDs (number)'
    }.issubset(df.columns):

        df['combined_risk'] = (
            df['Number of sexual partners'] *
            (df['Hormonal Contraceptives'] + 1) *
            (df['STDs (number)'] + 1)
        )

    return df