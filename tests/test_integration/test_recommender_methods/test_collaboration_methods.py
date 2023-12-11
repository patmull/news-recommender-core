import pandas as pd

from src.methods.user_based.collaboration_based_recommendation import SvdClass


def test_run_svd():
    test_user_id = 431
    svd = SvdClass()
    results = svd.run_svd(test_user_id)
    assert type(results) is dict
    results = svd.run_svd(test_user_id, dict_results=False)
    assert isinstance(results, pd.DataFrame)
    assert len(results.index) > 0
