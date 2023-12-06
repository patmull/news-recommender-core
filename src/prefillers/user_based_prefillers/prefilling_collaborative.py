import traceback

from src.custom_exceptions.exceptions import TestRunException
from src.prefillers.prefiller import UserBased

default_methods = ['svd', 'user_keywords', 'best_rated_by_others_in_user_categories', 'hybrid', 'hybrid_fuzzy']


def run_prefilling_collaborative(methods=None, user_id=None, test_run=False):
    """
    The main runner for prefilling user recommender articles
    @param methods: list of methods to use, needs to be from the domain of supported methods
    @param user_id: integer of user ID
    @param test_run: Set to true if this is test_run so it prevents the actual prefilling of value to database.
    This is an workaround and also for making sure that the value does not leak into a database.
    @return:
    """
    if methods is None:
        methods = default_methods
    else:
        if not set(methods).issubset(default_methods):
            raise ValueError("Methods parameter needs to be set to supported parameters " + str(default_methods))

    try:
        for method in methods:
            # Calling prefilling check_free_space_job user based..."
            user_based = UserBased()
            user_based.prefilling_job_user_based(method=method, db="pgsql", user_id=user_id, test_run=test_run,
                                                 skip_already_filled=False)
    except TestRunException as e:
        raise e
    except Exception as e:
        traceback.print_exception(None, e, e.__traceback__)
