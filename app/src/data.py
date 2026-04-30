import pandas as pd

from app.src.streamlit_helper import StreamlitHelper

class LoadData:

    def __init__(
        self, 
        path,
        targets,
        cols_remove
    ):
        self.path = path
        self.targets = targets
        self.cols_remove = cols_remove
        self.deleted_cols = self.targets + self.cols_remove

    def __handle_missing_values(self, dataset):
        print('Realizando tratamento inicial nos dados ...')
        ds = dataset.drop(columns=self.deleted_cols)
        return ds
    
    def get_dataset(self):
        print(
            f'Lendo o Dataset: {self.path}'
        )
        dataset = pd.read_csv(
            self.path,
            na_values='?'
        )
        return dataset

    def get_dataset_treated(self):
        pd.set_option('display.max_columns', None)
        dataset = self.get_dataset()
        dataset = self.__handle_missing_values(dataset)
        return dataset