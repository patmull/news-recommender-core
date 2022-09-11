from src.recommender_core.recommender_algorithms.content_based_algorithms.tfidf import TfIdf

word2vec_embedding = None
doc2vec_model = None
lda_model = None


# noinspection PyPep8
def main():
    # database = Database()
    # database.insert_posts_dataframe_to_cache() # for update

    # searched_slug = "zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy"
    # print(doc2vecClass.get_similar_doc2vec(slug))
    # searched_slug = "zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy"
    # searched_slug = "facr-o-slavii-a-rangers-verime-v-objektivni-vysetreni-odmitame-rasismus"
    # searched_slug = 'krasa-se-skryva-v-exotickem-ovoci-kosmetika-kterou-na-podzim-musite-mit'

    # STEMMING
    # word = "rybolovný"
    # # print(cz_stem(word))
    # # print(cz_stem(word,aggressive_stemming=True))

    # langdata = simplemma.load_data('cs')
    # # print(simplemma.lemmatize(word, langdata))
    # # print(tfidf.cz_lemma("nejnevhodnější"))

    # print(tfidf.preprocess_single_post("zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy",supplied_json=True))

    # gensim = GenSim()
    # gensim.get_recommended_by_slug("zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy")

    tfidf = TfIdf()
    # print(tfidf.recommend_posts_by_all_features_preprocessed('zdrazil-vam-dodavatel-elektrinu-nebo-plyn-brante-se-moznosti-je-nekolik'))
    tfidf.analyze('zdrazil-vam-dodavatel-elektrinu-nebo-plyn-brante-se-moznosti-je-nekolik')
    """
    print(tfidf.recommend_posts_by_all_features_preprocessed(searched_slug))
    print(tfidf.recommend_posts_by_all_features_preprocessed_with_full_text(searched_slug))
    """
    # print(tfidf.recommend_posts_by_all_features('sileny-cesky-plan-dva-roky-trenoval-ted-chce-sam-preveslovat-atlantik'))
    # print(tfidf.recommend_posts_by_all_features_preprocessed('sileny-cesky-plan-dva-roky-trenoval-ted-chce-sam-preveslovat-atlantik'))
    """
    tfidf = TfIdf()
    keywords = "fotbal hokej"
    print(tfidf.keyword_based_comparison(keywords))
    """

    """
    doc2vecClass = Doc2VecClass()
    print(doc2vecClass.get_similar_doc2vec(searched_slug,train_enabled=False))
    print(doc2vecClass.get_similar_doc2vec_with_full_text(searched_slug,train_enabled=False))

    lda = Lda()
    print("--------------LDA------------------")
    print(lda.get_similar_lda(searched_slug))
    print("--------------LDA FULL TEXT------------------")
    print(lda.get_similar_lda_full_text(searched_slug))
    """
    # lda = Lda()
    # print(lda.get_similar_lda('salah-pomohl-hattrickem-ztrapnit-united-soucek-byl-u-vyhry-nad-tottenhamem',
    # train_enabled=True, display_dominant_topics=True))
    # print(lda.get_similar_lda_full_text('salah-pomohl-hattrickem-ztrapnit-united-soucek-byl-u-vyhry-nad-tottenhamem',
    # train_enabled=False, display_dominant_topics=False))
    # lda.display_lda_stats()
    # lda.find_optimal_model_idnes(body_text_model=True)
    """
    word2vecClass = Word2VecClass()
    start = time.time()
    print(word2vecClass.get_similar_word2vec(searched_slug))
    end = time.time()
    print("Elapsed time: " + str(end - start))
    start = time.time()
    # print(word2vecClass.get_similar_word2vec_full_text(searched_slug))
    end = time.time()
    print("Elapsed time: " + str(end - start))
    # print(psutil.cpu_percent())
    # print(psutil.virtual_memory())  # physical memory usage
    # print('memory % used:', psutil.virtual_memory()[2])
    """
    # word2vec = Word2VecClass()
    # word2vec.prefilling_job(full_text=True, reverse=False, random_order=True)

    # word2vec = Word2VecClass()
    # word2vec.prefilling_job(full_text=True, reverse=False)

    # prefiller = PreFiller()
    # prefiller.prefilling_job("tfidf", "pgsql", full_text=False, reverse=False, random_order=True)
    # prefiller.prefilling_job("doc2vec", "pgsql", full_text=False, reverse=False, random_order=True)
    # prefiller.prefilling_job("lda", "pgsql", full_text=False, reverse=False, random_order=True)
    """
    prefiller.prefilling_job("tfidf", "pgsql", full_text=True, reverse=False, random_order=False)
    prefiller.prefilling_job("doc2vec", "pgsql", full_text=True, reverse=False, random_order=False)
    prefiller.prefilling_job("lda", "pgsql", full_text=True, reverse=False, random_order=False)
    """
    """
    h = hpy()
    print(h.heap())
    """


if __name__ == "__main__": main()
