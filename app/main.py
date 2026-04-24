#!/usr/bin/env python3


from app.src.data import LoadData
from app.src.streamlit_helper import StreamlitHelper
from app.src.train_test import ModelTestTraining    

import os


DATA = [
    'data',
    'risk_factors_cervical_cancer.csv'
]


TARGETS = [
    'Hinselmann',
    'Schiller',
    'Citology'
]


COLS_REMOVE = [
    'Dx:Cancer',
    'Dx:CIN',
    'Dx:HPV',
    'Dx'
]


def get_data_path():
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    parent_dir = os.path.dirname(script_dir)
    data_path = os.path.join(
        parent_dir,
        *DATA
    )
    return data_path


if __name__ == '__main__':
    ui_helper = StreamlitHelper()
    ui_helper.display_title('Cervical Cancer Risk Factors Dataset')
    ui_helper.display_divider()

    data_path = get_data_path()
    load_data = LoadData(
        path=data_path,
        targets=TARGETS,
        cols_remove=COLS_REMOVE
    )
    df = load_data.get_dataset()
    
    model_test_training = ModelTestTraining(
        dataset=df,
        target_column='Biopsy',
        random_state=42,
        test_size=0.2
    )
    model_test_training.train_test()