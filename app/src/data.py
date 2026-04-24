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

    def __handle_missing_values(self, dataset):
        pd.set_option('display.max_columns', None)
        dataset = dataset.drop(columns=self.targets + self.cols_remove)
        return dataset

    def get_dataset(self):
        self.ui_helper.display_markdown(
            f'**Lendo o dataset a partir do caminho:** {self.path}'
        )
        self.ui_helper.display_divider()
        dataset = pd.read_csv(
            self.path,
            na_values='?'
        )
        self.ui_helper.display_text(
            'Dataset Original com valores do tipo NA/NaN representados por "?" :'
        )
        self.ui_helper.display_dataframe(dataset)
        self.ui_helper.display_divider()
        dataset = self.__handle_missing_values(dataset)
        return dataset