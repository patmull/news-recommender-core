import re

py_file_path = '/home/patri/Documents/Codes/moje-clanky-core/news-recommender-core/research/relevance_statistics.py'

with open(py_file_path, 'r') as file:
    code_string = file.read()

commented_code = re.sub(r'print\("(.*)"\)', r'# \1', code_string)

with open(py_file_path, 'w') as file:
    file.write(commented_code)