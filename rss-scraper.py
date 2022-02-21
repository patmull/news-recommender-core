import json
import os
import xml.etree.ElementTree as ET
import urllib.request
import traceback
import ssl
import time
import mysql.connector
import schedule as scheduled
import feedparser
from slugify import slugify
import time

DB_USER = os.environ.get('DB_RECOMMENDER_USER')
DB_PASSWORD = os.environ.get('DB_RECOMMENDER_PASSWORD')
DB_NAME = os.environ.get('DB_RECOMMENDER_NAME')


# does not work yet
# xml.etree.ElementTree.ParseError: no element found: line 1, column 0
class XmlParser:
    ctx = None
    tree = None
    web_server = None
    url = None

    def __init__(self, url):
        self.url = url
        ssl._create_default_https_context = ssl._create_unverified_context
        self.ctx = ssl._create_unverified_context()

    def open_url(self):
        self.web_server = urllib.request.urlopen(self.url, context=self.ctx)
        self.tree = ET.parse(self.web_server)

    def getxml(self):

        resource = urllib.request.urlopen(self.url)

        try:
            data = resource.read().decode(resource.headers.get_content_charset())
        except:
            print("Failed to parse xml from response (%s)" % traceback.format_exc())
        return data

    def run_parser(self):

        self.open_url()

        self.tree = ET.parse(self.web_server)

        title = ""
        link = ""
        descripton = ""

        for post in self.tree.findall('.//channel/post'):
            idnes_article = Article()

            title = post.findall("title")
            link = post.findall("link")
            descripton = post.findall("description")

        print(title, link, descripton)


class RssFeedParser():
    url = None

    def __init__(self, url):
        self.url = url

    def run_parser(self):
        if hasattr(ssl, '_create_unverified_context'):
            ssl._create_default_https_context = ssl._create_unverified_context

        feed = feedparser.parse(self.url)
        print(len(feed.entries))

        cnx = mysql.connector.connect(user=DB_USER, password=DB_PASSWORD,
                                      host='eu-cdbr-west-03.cleardb.net',
                                      database=DB_NAME)

        cursor = cnx.cursor()

        database = Database(cnx, cursor)

        for post in feed.entries:
            article = Article()
            print("------")
            print(post.title)
            article.title = post.title
            # if 'subtitle' in post:
            #  print(post.link)
            print(post.published)
            article.published_date = post.published
            print(post.description)
            article.excerpt = post.description
            print(post.category)

            article.category = article.get_category(post.category)

            article.link = post.link

            article.body = post.description + "<p><a href='" + article.link + "'><b>Odkaz na původní článek</b></a></p>"
            print(article.body)

            # article.post_image = ???
            # print(post.media.content)

            database.store_post(article)


class Database:
    cnx = None
    cursor = None

    def __init__(self, cnx, cursor):
        self.cnx = cnx
        self.cursor = cursor

    def store_post(self, article):

        category_id_recognized = article.category

        query = ("SELECT COUNT(1) FROM posts WHERE title = '%s'" % (article.title))

        self.cursor.execute(query)

        if self.cursor.fetchone()[0]:
            print("Record exists.")
        else:
            print("Record doesn't exists.")
            slug = slugify(article.title)
            try:
                now = time.strftime('%Y-%m-%d %H:%M:%S')
                query = """INSERT INTO posts (`author_id`, `title`, `slug`, `excerpt`, `body`, `original_link`, `image`, `category_id`, `created_at`, `updated_at`, `published_at`) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"""
                inserted_values = (22, article.title, slug, article.excerpt, article.body, article.link, "default.jpg",
                                   category_id_recognized, now, now, now)
                self.cursor.execute(query, inserted_values)
                self.cnx.commit()

            except mysql.connector.Error as e:
                print("NOT INSERTED")
                print("Error code:", e.errno)  # error number
                print("SQLSTATE value:", e.sqlstate)  # SQLSTATE value
                print("Error message:", e.msg)  # error message
                print("Error:", e)  # errno, sqlstate, msg values
                s = str(e)
                print("Error:", s)  # errno, sqlstate, msg values
                self.cnx.rollback()


class Article:
    title = None
    link = None
    description = None
    published_date = None
    category = None
    image = None
    image_description = None
    post_image = None

    def get_category(self, category):

        category = category.lower()

        if 'ekonomika' in category:
            return 1
        elif any(s in category for s in (
        'ostrava', 'brno', 'karlovy vary', 'olomouc', 'pardubice', 'hradec', 'liberec', 'plzeň', 'jihlava',
        'budějovice', 'vary', 'labem')):
            return 2
        elif 'zprávy' and 'zahraničí' in category:
            return 3
        elif 'zprávy' and 'domácí' in category:
            return 4
        elif 'kultura' in category:
            return 5
        elif 'finance' in category:
            return 6
        elif 'móda' in category:
            return 7
        elif 'zdraví' in category:
            return 8
        elif 'vztahy' in category:
            return 9
        elif 'revue' in category:
            return 11
        elif 'mobil' in category:
            return 12
        elif any(s in category for s in ('věda', 'vesmír', 'technet')):
            return 13
        elif 'auto' in category:
            return 14
        elif 'hry' in category:
            return 15
        elif any(s in category for s in ('sport', 'fotbal', 'hokej')):
            return 17
        else:
            return 16


def job():
    url = "https://servis.idnes.cz/rss.aspx?c=zpravodaj"
    idnes_parser = RssFeedParser(url)
    idnes_parser.run_parser()


def main():
    scheduled.every(4).hours.do(job)
    #scheduled.every(30).seconds.do(job)

    while 1:
        scheduled.run_pending()
        time.sleep(1)


if __name__ == "__main__": main()
