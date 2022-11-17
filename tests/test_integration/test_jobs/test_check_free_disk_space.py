from unittest.mock import patch
from run_check_free_disk_space import disk_space_is_over_limit


def test_disk_space_is_over_limit_ok():
    with patch('shutil.disk_usage') as mocked_disk_usage:
        tested_size_ok = 800000000
        mocked_disk_usage().return_value = 0, 0, tested_size_ok
        assert disk_space_is_over_limit(tested_size_ok) is False

    with patch('shutil.disk_usage') as mocked_disk_usage_not_ok:
        tested_size_over_limit = 100000000
        mocked_disk_usage_not_ok().return_value = 0, 0, tested_size_over_limit
        assert disk_space_is_over_limit(tested_size_over_limit) is True
