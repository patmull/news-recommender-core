import pandas as pd
import pytest


# RUN WITH: python -m pytest tests/test_unit/test_hybrid_methods.py
from src.recommender_core.recommender_algorithms.hybrid_algorithms.hybrid_methods import select_list_of_posts_for_user
from src.recommender_core.recommender_algorithms.user_based_algorithms.user_relevance_classifier.classifier import \
    Classifier, get_df_predicted

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


def test_select_list_of_posts_for_user():
    test_user_id = 431
    searched_slug_1 = "zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy"
    searched_slug_2 = "salah-pomohl-hattrickem-ztrapnit-united-soucek-byl-u-vyhry-nad-tottenhamem"
    searched_slug_3 = "sileny-cesky-plan-dva-roky-trenoval-ted-chce-sam-preveslovat-atlantik"

    test_slugs = [searched_slug_1, searched_slug_2, searched_slug_3]
    list_of_slugs, list_of_slugs_from_history = select_list_of_posts_for_user(test_user_id, test_slugs)

    assert(
        type(list_of_slugs) is list,
        len(list_of_slugs) > 0,
        type(list_of_slugs_from_history) is list,
        len(list_of_slugs_from_history) > 0
     )