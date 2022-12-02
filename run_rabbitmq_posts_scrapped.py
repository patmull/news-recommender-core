import subprocess

filename = 'rabbitmq_consume_posts_scrapped.py'
while True:
    p = subprocess.Popen('python3 '+filename, shell=True).wait()

    """#if your there is an error from running 'my_python_code_A.py', 
    the while loop will be repeated, 
    otherwise the program will break from the loop"""
    if p != 0:
        continue
    else:
        break
