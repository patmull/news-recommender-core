import pandas as pd
import pytest

from src.recommender_core.recommender_algorithms.hybrid.classifier import Classifier, load_bert_model, get_df_predicted

# RUN WITH: python -m pytest tests/test_unit/test_hybrid_methods.py

classifier = Classifier()


def test_get_df_predicted():
    test_dict = {'col_1': ['test_1', 'test_2'], 'col2': ['test_3', 'test_4'], 'col_3': ['test_5', 'test_6']}
    df = pd.DataFrame(test_dict)
    target_variable_name = 'col2'
    df_predicted = get_df_predicted(df, target_variable_name='col2')
    assert target_variable_name in df_predicted.columns


@pytest.mark.parametrize("tested_input", [
    '',
    4,
    (),
    None,
    'ratings'
])
def test_svm_classifier_bad_relevance_by(tested_input):
    with pytest.raises(ValueError):
        svm = Classifier()
        assert svm.predict_relevance_for_user(use_only_sample_of=20, user_id=431, relevance_by=tested_input)


@pytest.mark.parametrize("tested_input", [
    '',
    15505661,
    (),
    None,
    'ratings'
])
def test_svm_classifier_bad_user_id(tested_input):
    with pytest.raises(ValueError):
        svm = Classifier()
        assert svm.predict_relevance_for_user(use_only_sample_of=20, user_id=tested_input, relevance_by='stars')


@pytest.mark.parametrize("tested_input", [
    '',
    (),
    None,
    'ratings'
])
def test_svm_classifier_bad_sample_number(tested_input):
    with pytest.raises(ValueError):
        svm = Classifier()
        assert svm.predict_relevance_for_user(use_only_sample_of=tested_input, user_id=431, relevance_by='stars')