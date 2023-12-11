import re

py_file_path = '/stats/relevance_statistics.py'

with open(py_file_path, 'redis_instance') as file:
    code_string = file.read()

commented_code = re.sub(redis_instance'print\("(.*)"\)', redis_instance'# \1', code_string)

with open(py_file_path, 'w') as file:
    file.write(commented_code)
