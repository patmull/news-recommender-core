import logging

import pytest

from src.prefillers.prefilling_all import check_needed_columns, run_prefilling
from src.recommender_core.data_handling.data_manipulation import DatabaseMethods


LOGGER = logging.getLogger(__name__)


@pytest.mark.integtest
def test_prefilling_all(caplog):
    database = DatabaseMethods()
    needed_columns = check_needed_columns(database)
    assert type(needed_columns) is list

    # Good Day
    # NOTICE: method "test_prefilled_all" corresponds to recommended_test_prefilled_all column which should contain prefilled
    LOGGER.info('Testing logger now.')
    run_prefilling(skip_cache_refresh=True, methods_short_text=["test_prefilled_all"], methods_full_text=[])
    assert 'Found 0 not prefilled posts in test_prefilled_all' in caplog.text

    """
    with pytest.raises(ValueError):
        # Bad Day (sort of)
        random_post_id = database.null_post_test_prefilled_record()
        # ** HERE WAS set_test_json_in_prefilled_records(random_post_id). ABANDONED DUE TO BETTER SCENARIO BELOW **
    """
    database.null_post_test_prefilled_record()
    run_prefilling(skip_cache_refresh=True, methods_short_text=["test_prefilled_all"], methods_full_text=[])
    assert 'Found 1 not prefilled posts in test_prefilled_all' in caplog.text

    # Here it fixes the problem
    run_prefilling(skip_cache_refresh=True, methods_short_text=["test_prefilled_all"], methods_full_text=[])
    # When running again, there should be 0 prefilled posts
    run_prefilling(skip_cache_refresh=True, methods_short_text=["test_prefilled_all"], methods_full_text=[])
    assert 'Found 0 not prefilled posts in test_prefilled_all' in caplog.text

