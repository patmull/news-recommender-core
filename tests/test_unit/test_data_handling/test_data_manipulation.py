import pytest

from src.recommender_core.data_handling.data_manipulation import DatabaseMethods


def test_singleton_cannot_be_instantiated_twice():
    DatabaseMethods()

    with pytest.raises(RuntimeError) as re:
        DatabaseMethods()
    assert str(re.value) == "Already instantiated!"
