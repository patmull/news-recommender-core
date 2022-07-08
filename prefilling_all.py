import traceback

from content_based_algorithms.prefiller import PreFiller
from data_connection import Database

prefiller = PreFiller()


# while True:
def run_prefilling():
    print("Running Tfidf short text")

    # TODO: Check needed columns
    # TODO: 'all_features_preprocessed' (probably every method relies on this)
    # TODO: 'keywords' (LDA but probably also other columns relies on this)
    # TODO: 'body_preprocessed' (LDA relies on this)

    database = Database()
    reverse = True
    random = False

    full_text = False

    method = "tfidf"
    prepare_and_run(database, method, full_text, reverse, random)

    method = "word2vec"
    prepare_and_run(database, method, full_text, reverse, random)

    method = "doc2vec"
    prepare_and_run(database, method, full_text, reverse, random)

    method = "lda"
    prepare_and_run(database, method, full_text, reverse, random)

    full_text = True

    method = "tfidf"
    prepare_and_run(database, method, full_text, reverse, random)

    method = "word2vec"
    prepare_and_run(database, method, full_text, reverse, random)

    method = "doc2vec"
    prepare_and_run(database, method, full_text, reverse, random)

    method = "lda"
    prepare_and_run(database, method, full_text, reverse, random)


def prepare_and_run(database, method, full_text, reverse, random):
    not_prefilled_posts = database.get_not_prefilled_posts(algorithm=method, full_text=full_text)
    print("Found " + str(len(not_prefilled_posts)) + " not prefilled posts")
    if len(not_prefilled_posts) > 0:
        try:
            prefiller.prefilling_job(method, "pgsql", full_text=full_text, reverse=reverse, random=random)
        except Exception as e:
            print("Exception occured " + str(e))
            print(traceback.print_exception(type(e), e, e.__traceback__))
    else:
        print("No not prefilled posts found")
        print("Skipping " + method + " full text")