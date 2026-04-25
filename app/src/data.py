import pandas as pd

from app.src.streamlit_helper import StreamlitHelper

class LoadData:

    def __init__(
        self, 
        path,
        targets,
        cols_remove
    ):
        self.ui_helper = StreamlitHelper()
        self.path = path
        self.targets = targets
        self.cols_remove = cols_remove
        self.deleted_cols = self.targets + self.cols_remove

    def __handle_missing_values(self, dataset):
        ds = dataset.drop(columns=self.deleted_cols)
        return ds
    
    def get_dataset(self):
        dataset = pd.read_csv(
            self.path,
            na_values='?'
        )
        return dataset

    def get_dataset_treated(self):
        pd.set_option('display.max_columns', None)
        self.ui_helper.display_markdown(
            f'**Lendo o dataset a partir do caminho:** {self.path}'
        )
        self.ui_helper.display_divider()

        dataset = self.get_dataset()

        self.ui_helper.display_text(
            'Dataset Original com valores do tipo NA/NaN representados por "?" :'
        )
        self.ui_helper.display_dataframe(dataset)
        self.ui_helper.display_divider()

        self.ui_helper.display_markdown(
            f'**Removendo as colunas de diagnóstico e os alvos do dataset...**: {self.deleted_cols}'
        )
        self.ui_helper.display_divider()

        self.ui_helper.display_text(
            'Dataset após tratamento dos valores faltantes:'
        )

        dataset = self.__handle_missing_values(dataset)

        self.ui_helper.display_dataframe(dataset)
        self.ui_helper.display_divider()

        return dataset