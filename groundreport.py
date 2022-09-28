import urllib
import urllib.request as request
from bs4 import BeautifulSoup
import ssl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

"""
def open_url(url):
   try:
       response = urllib.request.urlopen(url, timeout=10)
       return response
   except urllib.error.URLError as e:
       if isinstance(e.reason, socket.timeout):
           # handle timeout...
           pass
       raise e

"""

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36'}

homepage_url = "https://www.groundreport.com/"

ssl._create_default_https_context = ssl._create_unverified_context
ctx = ssl._create_unverified_context()

homepage = urllib.request.urlopen(homepage_url, context=ctx)
page_soup = BeautifulSoup(homepage, 'html.parser')
article_previews = page_soup.find_all('article')
# preview_text = article_preview.find_all('p')

for article_preview in article_previews:

    previews_headline = article_preview.find_all('h2')
    previews_link = article_preview.find_all('h2')
    previews_category_date = article_preview.find_all('p')
    previews_image = article_preview.find_all('img')
    previews_text = article_preview.find_all('div')

    print(previews_headline[0].get_text())
    print(previews_link[0].find('a')['href'])
    print(previews_category_date[0].get_text())
    print(previews_text[0].find('p').get_text())

    try:
        print(previews_image[0]['src'])
    except:
        print()
