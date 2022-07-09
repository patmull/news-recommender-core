from data_connection import Database


def test_prefilled_tfidf(full_text):
    database = Database()
    database.connect()
    posts = database.get_posts_with_no_prefilled_tfidf(full_text)
    return len(posts)


def test_prefilled_word2vec(full_text):
    database = Database()
    database.connect()
    posts = database.get_posts_with_no_prefilled_word2vec(full_text)
    return len(posts)


def test_prefilled_doc2vec(full_text):
    database = Database()
    database.connect()
    posts = database.get_posts_with_no_prefilled_doc2vec(full_text)
    return len(posts)


def test_prefilled_lda(full_text):
    database = Database()
    database.connect()
    posts = database.get_posts_with_no_prefilled_lda(full_text)
    return len(posts)


def test_prefilled_recommendations():
    number_of_tfidf = test_prefilled_tfidf(full_text=False)
    number_of_tfidf_full_text = test_prefilled_tfidf(full_text=True)
    number_of_word2vec = test_prefilled_word2vec(full_text=False)
    number_of_word2vec_full_text = test_prefilled_word2vec(full_text=True)
    number_of_doc2vec = test_prefilled_word2vec(full_text=False)
    number_of_doc2vec_full_text = test_prefilled_word2vec(full_text=True)
    number_of_lda = test_prefilled_word2vec(full_text=False)
    number_of_lda_full_text = test_prefilled_word2vec(full_text=True)

    assert number_of_tfidf == 0
    assert number_of_tfidf_full_text == 0
    assert number_of_word2vec == 0
    assert number_of_word2vec_full_text == 0
    assert number_of_doc2vec == 0
    assert number_of_doc2vec_full_text == 0
    assert number_of_lda == 0
    assert number_of_lda_full_text == 0