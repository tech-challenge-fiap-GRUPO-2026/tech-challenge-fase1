
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
# from sklearn.preprocessing import StandardScaler, MinMaxScaler

from app.src.pipeline_functions import (
    agrupar_colunas_preenchido,
    criar_atributos_estrategicos,
    # discretizar_demograficos_comportamentais,
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
    ):
        self.ui_helper = StreamlitHelper()
        self.dataset = dataset
        self.target_column = target_column
        self.random_state = random_state
        self.test_size = test_size
        self.pipeline = self.__build_pipeline()
        self.X = dataset.copy().drop(columns=[target_column])
        self.y = dataset.copy()[target_column]

    def __build_pipeline(self):
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
    
    def train_test(self):
        self.ui_helper.display_header('Treinamento e Teste do Modelo')
        self.ui_helper.display_divider()

        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            stratify=self.y, 
            random_state=self.random_state
        )

        self.ui_helper.display_markdown(
            f'**Valor do Random State:** {self.random_state} '
        )
        self.ui_helper.display_markdown(
            f'**Porcentagem utilizada para testes:** {"{:.2f}%".format(self.test_size * 100)} '
        )
        self.ui_helper.display_markdown(
            f'**Tamanho do conjunto de treinamento:** {len(X_train)} amostras'
        )
        self.ui_helper.display_markdown(
            f'**Tamanho do conjunto de teste:** {len(X_test)} amostras'
        )
        self.ui_helper.display_divider()

        self.ui_helper.display_markdown("Dados de Treinamento:")
        self.ui_helper.display_dataframe(X_train)

        self.ui_helper.display_markdown("Dados de Teste:")
        self.ui_helper.display_dataframe(X_test)

        self.ui_helper.display_divider()
        self.ui_helper.display_subheader("Etapas do Pipeline:")
        self.ui_helper.display_json(self.pipeline.named_steps)
        self.ui_helper.display_divider()

        self.pipeline.fit(
            X_train,
            y_train
        )

        y_pred = self.pipeline.predict(X_test)
        report = classification_report(y_test, y_pred)

        self.ui_helper.display_subheader("Relatório de Classificação:")
        self.ui_helper.display_code(report, wrap_lines=True)