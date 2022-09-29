import traceback

from src.custom_exceptions.exceptions import TestRunException
from src.prefillers.prefiller import UserBased

default_methods = ['svd', 'user_keywords', 'best_rated_by_others_in_user_categories']


def run_prefilling_collaborative(methods=None, user_id=None, test_run=False):
    # db_columns = ['svd, 'user_keywords'] # NOTICE: Needs to correspond with DB column names
    # db_columns = ['user_keywords']
    if methods is None:
        methods = default_methods
    else:
        if not set(methods).issubset(default_methods):
            raise ValueError("Methods parameter needs to be set to supported parameters " + str(default_methods))

    # while True: # this can force of run no matter what
    try:
        for method in methods:
            print("Calling prefilling job user based...")
            user_based = UserBased()
            user_based.prefilling_job_user_based(method=method, db="pgsql", user_id=user_id, test_run=test_run,
                                                 skip_already_filled=False)
    except TestRunException as e:
        raise e
    except Exception as e:
        print("Exception occurred " + str(e))
        traceback.print_exception(None, e, e.__traceback__)
