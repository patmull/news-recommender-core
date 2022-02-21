import time

import redis
from urllib.parse import urlparse

url = urlparse('redis://default:RuTYjOqZZofckHAPTNqUMlg8XjT1nQdZ@redis-18878.c59.eu-west-1-2.ec2.cloud.redislabs.com:18878')
r = redis.Redis(host=url.hostname, port=url.port, username=url.username, password=url.password)

"""
r.set("TestKey", "TestValue")
r.set("TestKey2", "TestValue2")

france_capital = r.get("TestKey")
german_capital = r.get("TestKey2")

print(france_capital)
print(german_capital)

r.mset({'slug': 'chaos-v-mapach-kavkazu-proc-armenie-a-azerbajdzan-nebojuji-jen-o-karabach&content-based-method', 'coefficient': '0.72'})

if(r.exists('chaos-v-mapach-kavkazu-proc-armenie-a-azerbajdzan-nebojuji-jen-o-karabach&content-based-method')):
    print(r.get('chaos-v-mapach-kavkazu-proc-armenie-a-azerbajdzan-nebojuji-jen-o-karabach&content-based-method'))
else:
    print('Sorry. Cannot find the post.')

r.psetex('slug',4000,'chaos-v-mapach-kavkazu-proc-armenie-a-azerbajdzan-nebojuji-jen-o-karabach') # limited time of storage
print(r.get('slug'))

time.sleep(10)

print(r.get('slug'))
"""
r.set("key_from_python","value_from_python")
print(r.get("key_from_python"))
