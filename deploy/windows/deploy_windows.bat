git fetch
git pull
call %~dp0venv\Scripts\activate
python -m pytest tests/test_integration/test_only_local_working_methods
tox tests/test_integration/test_only_local_working_methods
