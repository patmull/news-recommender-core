from research.user_based.user_relevance_eval import user_relevance_asessment
from src.constants.naming import Naming
from src.recommender_core.data_handling.data_manipulation import get_redis_connection

REDIS_TOP_FOLDER = 'statistics'


def prepare_redis():
    testing_redis_key = REDIS_TOP_FOLDER + Naming.REDIS_DELIMITER + 'testing-redis-key'
    r = get_redis_connection()
    try:
        r.delete(testing_redis_key)
    except Exception as e:
        print("Redis delete exception:")
        print(e)

    return r, testing_redis_key


def test_user_eval():
    r, testing_redis_key = prepare_redis()
    assert (r.exists(testing_redis_key) == 0)
    user_relevance_asessment(save_to_redis=True)
    assert (r.exists(testing_redis_key) > 0)
    assert r.get(REDIS_TOP_FOLDER + ':testing-redis-key') == 'testing-redis-value'




