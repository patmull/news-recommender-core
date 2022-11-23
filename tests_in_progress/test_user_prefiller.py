from unittest.mock import Mock, patch

from src.prefillers.user_based_prefillers.prefilling_collaborative import run_prefilling_collaborative


class CallException(Exception):
    pass


m = Mock(side_effect=CallException('Function called!'))


# TODO: Finish or erase this!
def caller_test():
    run_prefilling_collaborative()
    raise RuntimeError("This should not be called!")


@patch("src.prefillers.user_based_prefillers.prefilling_collaborative.run_prefilling_collaborative", m)
def test_called():
    try:
        run_prefilling_collaborative()
    except CallException:
        print("Called!")
        return
    assert "Exception not called!"


if __name__ == "__main__":
    test_called()
