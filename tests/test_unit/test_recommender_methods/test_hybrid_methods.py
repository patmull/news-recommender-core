import pandas as pd

from src.recommender_core.recommender_algorithms.hybrid.classifier import Classifier, load_bert_model, get_df_predicted

# RUN WITH: python -m pytest tests/unit/test_hybrid_methods.py

classifier = Classifier()


def test_get_df_predicted():
    test_dict = {'col_1': ['test_1', 'test_2'], 'col2': ['test_3', 'test_4'], 'col_3': ['test_5', 'test_6']}
    df = pd.DataFrame(test_dict)
    target_variable_name = 'col2'
    df_predicted = get_df_predicted(df, target_variable_name='col2')
    assert target_variable_name in df_predicted.columns
