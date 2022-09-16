import traceback

from src.recommender_core.recommender_algorithms.content_based_algorithms.prefiller import prefilling_job_user_based


def run_prefilling_svd():
    # methods = ['svd, 'user_keywords']
    methods = ['user_keywords']  # NOTICE: Needs to correspond with DB column names
    # while True:
    try:
        for method in methods:
            prefilling_job_user_based(method=method, db="pgsql")
    except Exception as e:
        print("Exception occurred " + str(e))
        traceback.print_exception(None, e, e.__traceback__)
