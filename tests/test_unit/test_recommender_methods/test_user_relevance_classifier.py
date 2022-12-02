import pytest

from src.recommender_core.recommender_algorithms.user_based_algorithms.user_relevance_classifier.classifier import \
    Classifier


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
