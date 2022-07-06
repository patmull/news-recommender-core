import traceback

from content_based_algorithms.prefiller import PreFiller
from data_connection import Database

prefiller = PreFiller()


# while True:
def run_prefilling():
    print("Running Tfidf short text")
    database = Database()
    method = "tfidf"
    full_text = False
    not_prefilled_posts = database.get_not_prefilled_posts(algorithm=method, full_text=full_text)
    print("Found " + str(len(not_prefilled_posts)) + " not prefilled posts")
    if len(not_prefilled_posts) > 0:
        try:
            prefiller.prefilling_job("tfidf", "pgsql", full_text=False, reverse=True, random=True)
        except Exception as e:
            print("Exception occured " + str(e))
            print(traceback.print_exception(type(e), e, e.__traceback__))
    else:
        print("No not prefilled posts found")
        print("Skipping " + method + " short text.")

    method = "doc2vec"
    full_text = False
    not_prefilled_posts = database.get_not_prefilled_posts(algorithm=method, full_text=full_text)
    print("Found " + str(len(not_prefilled_posts)) + " not prefilled posts")
    if len(not_prefilled_posts) > 0:
        try:
            print("Running Doc2Vec short text")
            prefiller.prefilling_job("doc2vec", "pgsql", full_text=False, reverse=True, random=True)
        except Exception as e:
            print("Exception occured " + str(e))
            print(traceback.print_exception(type(e), e, e.__traceback__))
    else:
        print("No not prefilled posts found")
        print("Skipping " + method + " short text")

    method = "lda"
    full_text = False
    not_prefilled_posts = database.get_not_prefilled_posts(algorithm=method, full_text=full_text)
    print("Found " + str(len(not_prefilled_posts)) + " not prefilled posts")
    if len(not_prefilled_posts) > 0:
        try:
            print("Running LDA short text prefilling")
            prefiller.prefilling_job("lda", "pgsql", full_text=False, reverse=True, random=True)
        except Exception as e:
            print("Exception occured " + str(e))
            print(traceback.print_exception(type(e), e, e.__traceback__))
    else:
        print("No not prefilled posts found")
        print("Skipping " + method + " short text")

    method = "doc2vec"
    full_text = True
    not_prefilled_posts = database.get_not_prefilled_posts(algorithm=method, full_text=full_text)
    print("Found " + str(len(not_prefilled_posts)) + " not prefilled posts")
    if len(not_prefilled_posts) > 0:
        try:
            print("Running TfIdf full text")
            prefiller.prefilling_job("tfidf", "pgsql", full_text=True, reverse=True, random=True)
        except Exception as e:
            print("Exception occured " + str(e))
            print(traceback.print_exception(type(e), e, e.__traceback__))
    else:
        print("No not prefilled posts found")
        print("Skipping " + method + " full text")

    method = "doc2vec"
    full_text = True
    not_prefilled_posts = database.get_not_prefilled_posts(algorithm=method, full_text=full_text)
    print("Found " + str(len(not_prefilled_posts)) + " not prefilled posts")
    if len(not_prefilled_posts) > 0:
        try:
            print("Running Doc2Vec full text")
            prefiller.prefilling_job("doc2vec", "pgsql", full_text=True, reverse=True, random=True)
        except Exception as e:
            print("Exception occured " + str(e))
            print(traceback.print_exception(type(e), e, e.__traceback__))
    else:
        print("No not prefilled posts found")
        print("Skipping " + method + " full text")

    method = "lda"
    full_text = True
    not_prefilled_posts = database.get_not_prefilled_posts(algorithm=method, full_text=full_text)
    print("Found " + str(len(not_prefilled_posts)) + " not prefilled posts")
    if len(not_prefilled_posts) > 0:
        try:
            print("Running LDA full text")
            prefiller.prefilling_job("lda", "pgsql", full_text=True, reverse=True, random=True)
        except Exception as e:
            print("Exception occured " + str(e))
            print(traceback.print_exception(type(e), e, e.__traceback__))
    else:
        print("No not prefilled posts found")
        print("Skipping " + method + " full text")

    method = "word2vec"
    full_text = False
    not_prefilled_posts = database.get_not_prefilled_posts(algorithm=method, full_text=full_text)
    if len(not_prefilled_posts) > 0:
        try:
            print("Running Word2Vec short text")
            prefiller.prefilling_job("word2vec", "pgsql", full_text=False, reverse=True, random=True)
        except Exception as e:
            print("Exception occured " + str(e))
            print(traceback.print_exception(type(e), e, e.__traceback__))
    else:
        print("No not prefilled posts found")
        print("Skipping " + method + " short text")

    method = "word2vec"
    full_text = True
    not_prefilled_posts = database.get_not_prefilled_posts(algorithm=method, full_text=full_text)
    if len(not_prefilled_posts) > 0:
        try:
            print("Running Word2Vec full text")
            prefiller.prefilling_job("word2vec", "pgsql", full_text=True, reverse=True, random=True)
        except Exception as e:
            print("Exception occured " + str(e))
            print(traceback.print_exception(type(e), e, e.__traceback__))
    else:
        print("No not prefilled posts found")
        print("Skipping " + method + " full text")
