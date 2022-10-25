from datetime import datetime, timezone, timedelta

import pandas as pd

from src.recommender_core.recommender_algorithms.hybrid_algorithms.hybrid_methods import boost_by_article_freshness, \
    HybridConstants


def test_boost_by_freshness():
    coeff_1 = 0.5
    coeff_2 = 0.8
    coeff_3 = 0.6
    coeff_4 = 0.856
    now = datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    fresh = now + datetime.timedelta(minutes=-30)
    older_than_hour = now + timedelta(hours=-3)
    older_than_day = now + timedelta(days=-3)
    older_than_5_days = now + timedelta(days=-6)
    tested_data = {
        'coefficient': [coeff_1, coeff_2, coeff_3, coeff_4],
        'post_created_at': [fresh, older_than_hour, older_than_day, older_than_5_days]
    }
    tested_df = pd.DataFrame(tested_data)
    tested_df = boost_by_article_freshness(tested_df)

    hybrid_constants = HybridConstants()
    assert tested_df.iloc[0]['coefficient'] == coeff_1 * hybrid_constants.coeff_and_hours_1[0]
    assert tested_df.iloc[1]['coefficient'] == coeff_2 * hybrid_constants.coeff_and_hours_2[0]
    assert tested_df.iloc[2]['coefficient'] == coeff_3 * hybrid_constants.coeff_and_hours_3[0]
    assert tested_df.iloc[3]['coefficient'] == coeff_4 * hybrid_constants.coeff_and_hours_4[0]
