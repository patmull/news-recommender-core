from datetime import datetime

import pytest

from src.recommender_core.data_handling.data_manipulation import get_redis_connection


class RedisTest:

    @pytest.mark.integtest
    def test_redis(self):
        r = get_redis_connection()
        now = datetime.now()
        test_value = 'test_' + str(now.strftime("%m/%d/%Y %H:%M:%S"))
        r.set('test_pair', test_value)
        assert r.get('test_pair').decode() == test_value

    # RUN WITH: tests/test_integration/test_data_handling/test_data_queries.py::test_redis_values
    @pytest.mark.integtest
    def test_redis_values(self):
        r = get_redis_connection()
        now = datetime.now()
        test_value = 'test_' + str(now.strftime("%m/%d/%Y %H:%M:%S"))
        r.delete(test_value)
        r.set('test_pair', test_value)
        assert r.get('test_pair').decode() == test_value
        r.delete("posts_by_pred_ratings_user_371")
        test_value = "vytvorili-prvni-rib-eye-steak-ze-zkumavky-chutna-jako-prave-maso"
        # TODO: Restrict PgSQL from ever deleting this user
        test_user_key = "posts_by_pred_ratings_user_371"
        r.delete(test_user_key)
        for key in test_user_key:
            r.delete(key)
        r.sadd(test_user_key, '')
        print("r.smembers(test_value):")
        print(r.smembers(test_user_key))
        res = r.sadd(test_user_key, test_value)
        print(res)
        print(r.smembers(test_user_key))
        assert res == 1
        assert type(r.smembers(test_user_key)) == set
