
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

from app.src.constants import (
    MODEL_FILE,
    X_TEST_FILE,
    Y_TEST_FILE
)
from app.src.pipeline_functions import (
    agrupar_colunas_preenchido,
    criar_atributos_estrategicos,
    reduzir_redundancia_smokes,
    remover_colunas_sem_variacao,
    remover_coluna_std_tempo,
    tratar_missing_com_indicador,
    transformar_contraceptivos,
    transformar_demograficos_comportamentais, 
)
from app.src.streamlit_helper import StreamlitHelper


class ModelTestTraining(object):

    def __init__(
        self,
        dataset,
        target_column,
        random_state,
        test_size,
        deploy_dir_path
    ):
        self.dataset = dataset
        self.target_column = target_column
        self.random_state = random_state
        self.test_size = test_size
        self.deploy_dir_path = deploy_dir_path
        self.pipeline = self.__build_pipeline()
        self.X = dataset.copy().drop(columns=[target_column])
        self.y = dataset.copy()[target_column]

    def __build_pipeline(self):
        print('Criando o Pipeline ...')
        pipeline = Pipeline([
            ('remover', FunctionTransformer(remover_colunas_sem_variacao)),
            ('faltantes', FunctionTransformer(tratar_missing_com_indicador)),
            ('agrupar', FunctionTransformer(agrupar_colunas_preenchido)),
            ('tabagismo_redundancia', FunctionTransformer(reduzir_redundancia_smokes)),
            ('transformar_demo_comp', FunctionTransformer(transformar_demograficos_comportamentais)),
            ('contraceptivos', FunctionTransformer(transformar_contraceptivos)),
            ('remover_std_tempo', FunctionTransformer(remover_coluna_std_tempo)),
            ('novos_atributos', FunctionTransformer(criar_atributos_estrategicos)),
            ('model', DecisionTreeClassifier(
                class_weight={0: 1, 1: 10},
                max_depth=4,
                min_samples_leaf=5,
                random_state=self.random_state
            ))
        ])
        return pipeline

    def __deploy(self, X_test, y_test):
        print('Iniciando o Deploy do modelo ...')
        with open(
            os.path.join(self.deploy_dir_path, MODEL_FILE),
            'wb'
        ) as f:
            pickle.dump(self.pipeline, f)
        print('Iniciando o Deploy dos dados de Teste de X ...')
        with open(
            os.path.join(self.deploy_dir_path, X_TEST_FILE),
            'wb'
        ) as f:
            pickle.dump(X_test, f)
        print('Iniciando o Deploy dos dados de Teste de y ...')
        with open(
            os.path.join(self.deploy_dir_path, Y_TEST_FILE),
            'wb'
        ) as f:
            pickle.dump(y_test, f)
    
    def train_test(self, deploy=True):
        print('Separando os dados de treino e testes do modelo ...')
        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            stratify=self.y, 
            random_state=self.random_state
        )
        print('Iniciando o Treinamento do modelo ...')
        self.pipeline.fit(
            X_train,
            y_train
        )
        print('Iniciando o Teste do modelo ...')
        y_pred = self.pipeline.predict(X_test)
        report = classification_report(y_test, y_pred)
        print(report)
        if deploy:
            self.__deploy(X_test, y_test)
        